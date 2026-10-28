"""Known-target checks for frozen robust/recovery proposals."""
import unittest
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from selflensing.fitting.adaptation import tune_proposals, proposal_step
from selflensing.fitting.recovery import mixture_candidate, mixture_log_density, select_centres


def sample(target, warm, probability=0., steps=6000):
    frozen=np.asarray(warm.transform).copy()
    def step(state,key):
        p,score=state
        pk,ak=jax.random.split(key)
        if probability:
            y,c,_,r=mixture_candidate(pk,p,warm.transform,True,warm.recovery_centres,warm.recovery_transform,probability)
        else:
            delta,_=proposal_step(pk,p,warm.transform,True)
            y,c,r=p+delta,0.,jnp.zeros(score.shape,dtype=bool)
        sy=target(y)
        a=jnp.log(jax.random.uniform(ak,score.shape)) < sy-score+c
        p=jnp.where(a[...,None],y,p)
        score=jnp.where(a,sy,score)
        return (p,score),(p,r,a)
    _,out=jax.jit(lambda:jax.lax.scan(step,(warm.positions,target(warm.positions)),jax.random.split(jax.random.PRNGKey(73),steps)))()
    np.testing.assert_array_equal(warm.transform,frozen)
    return [np.asarray(x) for x in out]


class RecoveryTest(unittest.TestCase):
    def test_covariance_selection_uses_physical_scores(self):
        initial=np.random.default_rng(9).normal(size=(1,16,2))
        initial[0,-1]=[100.,100.]
        target=lambda p: -.5*jnp.sum(p*p,axis=-1)
        # A fake coordinate Jacobian makes the bad walker look competitive.
        transformed=lambda p: target(p)+jnp.where(p[...,0]>50.,10000.,0.)
        physical=lambda p,s: s-jnp.where(p[...,0]>50.,10000.,0.)
        warm=tune_proposals(transformed,initial,[.1,.1],steps=40,rounds=2,
            covariance=True,robust_covariance=True,score_transform=physical)
        self.assertLess(float(jnp.max(jnp.abs(warm.transform))),5.)
        self.assertLess(float(np.max(warm.eligible_fraction)),1.)

    def test_multiple_targets_have_independent_recovery_kernels(self):
        means=jnp.array([[1.,-2.],[-3.,4.]])
        widths=jnp.array([[.1,2.],[2.,.2]])
        target=lambda p: -.5*jnp.sum(((p-means[:,None])/widths[:,None])**2,axis=-1)
        initial=means[:,None]+jax.random.normal(jax.random.PRNGKey(6),(2,32,2))*widths[:,None]
        warm=tune_proposals(target,initial,[.3,.3],steps=800,covariance=True,
            robust_covariance=True,recovery_probability=.1)
        chain,_,_=sample(target,warm,.1,steps=3000)
        np.testing.assert_allclose(chain[500:].mean(axis=(0,2)),means,atol=.1)
        np.testing.assert_allclose(chain[500:].std(axis=(0,2)),widths,rtol=.1)

    def test_density_ratio_matches_scipy(self):
        from scipy.stats import multivariate_t
        L=np.array([[.8,0.],[.4,1.2]])
        centres=np.array([[-4.,0.],[1.,2.],[1.,2.]])
        x=np.array([[-2.,3.],[4.,-1.]])
        expected=np.log(np.mean([multivariate_t.pdf(x,loc=c,shape=L@L.T,df=3) for c in centres],axis=0))
        got=np.asarray(mixture_log_density(jnp.array(x[None]),jnp.array(centres[None]),jnp.array(L[None])))[0]
        np.testing.assert_allclose(got-got[0],expected-expected[0],atol=1e-12)

    def test_diverse_pool_keeps_good_modes_and_excludes_bad(self):
        positions=jnp.array([[[-4.,0.],[-3.9,0.],[4.,0.],[4.1,0.],[200.,200.]]])
        pool=np.asarray(select_centres(positions,jnp.array([[0.,-.2,-1.,-1.2,-80.]]),jnp.ones((1,2)),count=4))
        self.assertTrue((pool[...,0]<0).any() and (pool[...,0]>0).any())
        self.assertLess(np.abs(pool).max(),5.)

    def test_unequal_modes_with_deliberately_wrong_proposal_weights(self):
        from selflensing.fitting.adaptation import WarmupResult
        centres=jnp.array([-4.,4.])
        def target(p):
            return logsumexp(jnp.log(jnp.array([.3,.7]))-.5*((p[...,0,None]-centres)/.5)**2,axis=-1)
        # 15:1 proposal weights, whereas the posterior is 3:7.
        pool=jnp.array(([-4.]*15+[4.])).reshape(1,16,1)
        initial=np.tile(np.array([-4.,4.]),32).reshape(1,64,1)
        warm=WarmupResult(jnp.array(initial),jnp.array([[[.5]]]),np.empty(0),pool,jnp.array([[[.5]]]))
        chain,_,_=sample(target,warm,.2,steps=12000)
        fraction=float((chain[2000:,...,0]>0).mean())
        print('Unequal mode right mass',fraction,flush=True)
        self.assertAlmostEqual(fraction,.7,delta=.035)
        for side,centre in [(False,-4.),(True,4.)]:
            x=chain[2000:,...,0]; x=x[(x>0)==side]
            self.assertAlmostEqual(float(x.mean()),centre,delta=.05)
            self.assertAlmostEqual(float(x.std()),.5,delta=.05)

    def test_factorial_gaussian_and_bad_initial_walker(self):
        cov=np.array([[1.,.9],[.9,1.]])
        inv=jnp.asarray(np.linalg.inv(cov))
        def target(p):
            main=-.5*jnp.einsum('...i,ij,...j->...',p,inv,p)
            bad=-80.-.5*jnp.sum(((p-jnp.array([200.,-200.]))/10.)**2,axis=-1)
            return jnp.logaddexp(main,bad)
        initial=np.random.default_rng(5).multivariate_normal([0.,0.],cov,size=(1,32))
        initial[0,-1]=[200.,-200.]
        rows=[]
        for robust in (False,True):
            for probability in (0.,.1):
                warm=tune_proposals(target,initial,[.3,.3],steps=1600,rounds=16,covariance=True,
                    robust_covariance=robust,recovery_probability=probability)
                chain,r,a=sample(target,warm,probability,steps=6000)
                kept=chain[1000:].reshape(-1,2)
                good=np.linalg.norm(kept,axis=1)<20
                row=dict(robust=robust,recovery=probability,bad_fraction=float(1-good.mean()),
                    mean=kept.mean(axis=0).tolist(),cov=np.cov(kept.T).tolist(),acceptance=float(a.mean()))
                rows.append(row)
                print(row,flush=True)
                if probability:
                    self.assertGreater(good.mean(),.999)
                    np.testing.assert_allclose(kept.mean(axis=0),0.,atol=.13)
                    np.testing.assert_allclose(np.cov(kept.T),cov,atol=.18)
                if robust:
                    np.testing.assert_allclose(kept[good].mean(axis=0),0.,atol=.13)
                    np.testing.assert_allclose(np.cov(kept[good].T),cov,atol=.18)
                    self.assertGreater(np.asarray(warm.eligible_fraction).min(),.8)

if __name__=='__main__': unittest.main()
