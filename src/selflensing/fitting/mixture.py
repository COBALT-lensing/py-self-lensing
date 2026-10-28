"""Frozen local-mixture independence Metropolis sampling."""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from scipy.special import gammaln


@dataclass
class LocalMixture:
    weights: np.ndarray
    centres: np.ndarray
    transforms: np.ndarray
    df: float = 5.

    def __post_init__(self):
        self.weights=np.asarray(self.weights,dtype=float)
        self.centres=np.asarray(self.centres,dtype=float)
        self.transforms=np.asarray(self.transforms,dtype=float)
        k,d=self.centres.shape
        if self.weights.shape!=(k,) or self.transforms.shape!=(k,d,d) or k<1:
            raise ValueError('Mixture weights, centres and transforms have incompatible shapes')
        if not all(np.isfinite(x).all() for x in (self.weights,self.centres,self.transforms)):
            raise ValueError('Mixture parameters must be finite')
        if np.any(self.weights<=0) or not np.isclose(self.weights.sum(),1.):
            raise ValueError('Mixture weights must be positive and sum to one')
        if not np.isfinite(self.df) or self.df<=0:
            raise ValueError('Student-t degrees of freedom must be positive')
        if np.any(np.diagonal(self.transforms,axis1=-2,axis2=-1)<=0) or not np.allclose(self.transforms,np.tril(self.transforms)):
            raise ValueError('Mixture transforms must be lower triangular with positive diagonal')

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            return cls(z['weights'],z['centres'],z['transforms'],float(z['df']))

    @classmethod
    def fit(cls, samples, components=64, seed=0, df=5., max_samples=60000):
        from sklearn.mixture import GaussianMixture
        from threadpoolctl import threadpool_limits
        samples=np.asarray(samples)
        rng=np.random.default_rng(seed)
        if len(samples)>max_samples:
            samples=samples[rng.choice(len(samples),max_samples,replace=False)]
        mean=samples.mean(axis=0); scale=samples.std(axis=0)
        if np.any(scale<=0):raise ValueError('Proposal training requires nonconstant dimensions')
        with threadpool_limits(limits=4):
            model=GaussianMixture(n_components=components,covariance_type='full',
                reg_covar=1e-7,max_iter=150,n_init=1,random_state=seed,tol=2e-3).fit((samples-mean)/scale)
        centres=model.means_*scale+mean
        cov=model.covariances_*scale[None,:,None]*scale[None,None,:]
        return cls(model.weights_,centres,np.linalg.cholesky(cov),df)

    def logpdf(self, position):
        position=jnp.asarray(position)
        delta=position[...,None,:]-jnp.asarray(self.centres)
        # Explicit per-component whitening; all component normalizers matter.
        inverse=jnp.asarray(np.linalg.inv(self.transforms))
        white=jnp.einsum('kij,...kj->...ki',inverse,delta)
        d=position.shape[-1]
        normalizer=gammaln((self.df+d)/2)-gammaln(self.df/2)-.5*d*np.log(self.df*np.pi)
        determinant=jnp.sum(jnp.log(jnp.diagonal(jnp.asarray(self.transforms),axis1=-2,axis2=-1)),axis=-1)
        kernel=normalizer-determinant-.5*(self.df+d)*jnp.log1p(jnp.sum(white**2,axis=-1)/self.df)
        return logsumexp(kernel+jnp.log(jnp.asarray(self.weights)),axis=-1)

    def draw(self, rng, shape):
        component=rng.choice(len(self.weights),size=shape,p=self.weights)
        noise=rng.normal(size=(*shape,self.centres.shape[-1]))
        noise*=np.sqrt(self.df/rng.chisquare(self.df,size=shape))[...,None]
        return self.centres[component]+np.einsum('...ij,...j->...i',self.transforms[component],noise)

    def save(self,path):
        np.savez_compressed(path,weights=self.weights,centres=self.centres,transforms=self.transforms,df=self.df)


def independence_scan(initial, initial_logp, initial_logq, proposals, logp, logq, log_uniform):
    """Precomputed iid proposals allow parallel likelihoods and exact serial MH."""
    def step(state, inputs):
        x,p,q=state;y,py,qy,u=inputs
        accept=u < py-p+q-qy
        x=jnp.where(accept[...,None],y,x)
        p=jnp.where(accept,py,p);q=jnp.where(accept,qy,q)
        return (x,p,q),(x,p,accept)
    return jax.lax.scan(step,(initial,initial_logp,initial_logq),
                        (proposals,logp,logq,log_uniform))


def sample_independence(target, proposal, initial, *, steps, seed, chunk=512, report=None):
    """Frozen proposal; retain all states, including rejections, without thinning."""
    if int(steps)!=steps or steps<1 or int(chunk)!=chunk or chunk<1:
        raise ValueError('steps and chunk must be positive integers')
    initial=np.asarray(initial)
    if initial.ndim!=2 or not np.isfinite(initial).all():
        raise ValueError('Initial positions must be a finite (walker, parameter) array')
    rng=np.random.default_rng(seed)
    x=jnp.asarray(initial);p=target(x);q=proposal.logpdf(x)
    density=jax.jit(proposal.logpdf);scan=jax.jit(independence_scan)
    xs=[];ps=[];accepted=[]
    walkers=len(initial)
    for start in range(0,steps,chunk):
        n=min(chunk,steps-start)
        y=proposal.draw(rng,(n,walkers));flat=y.reshape(-1,y.shape[-1])
        py=target(jnp.asarray(flat)).reshape(n,walkers)
        qy=density(jnp.asarray(flat)).reshape(n,walkers)
        (x,p,q),out=scan(x,p,q,jnp.asarray(y),py,qy,jnp.log(jnp.asarray(rng.uniform(size=(n,walkers)))))
        xs.append(np.asarray(out[0]));ps.append(np.asarray(out[1]));accepted.append(np.asarray(out[2]))
        if report: report(start+n,steps,float(np.mean(accepted[-1])))
    return np.concatenate(xs),np.concatenate(ps),np.concatenate(accepted)
