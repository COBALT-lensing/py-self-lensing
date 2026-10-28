"""Absolute Gaussian log-likelihood scaling, priors, and native GP preservation."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm
from selflensing.fitting.likelihood import (combined_chi2,numeric_combined_chi2,
    make_combined_chi2_vec,make_combined_chi2_vec_jax,
    make_batched_combined_chi2_vec_jax,make_batched_numeric_combined_chi2_vec_jax)
from selflensing.fitting.numeric import parameter_row,deterministic_models,physical_properties


class GaussianLikelihoodTest(unittest.TestCase):
    def setUp(self):
        self.p=jnp.array([1.,1.5707,9.,.2,.4])
        self.t=jnp.linspace(-.2,.2,4)
        self.lc,self.rv=deterministic_models(self.t,self.t,parameter_row(self.p,1.,1.),0.)
        self.err=jnp.array([.01,.02,.015,.01])
        self.rv_err=jnp.array([10.,20.,15.,10.])

    def args(self,photometry=True,rv=True):
        return dict(Mcomp=1.,Rcomp=1.,tau_eff=None,peak_time=0.,
            rv_times=self.t if rv else None,observed_rv=self.rv+self.rv_err*jnp.array([1.,-2.,.5,3.]) if rv else None,
            observed_rv_err=self.rv_err if rv else None,
            lc_times=self.t if photometry else None,observed_lc=self.lc+self.err*jnp.array([1.,-2.,.5,3.]) if photometry else None,
            observed_lc_err=self.err if photometry else None,detected_periods=jnp.array([]))

    def expected(self,args,profile=False):
        result=0.
        for key,model,error,do_profile in [('observed_lc',self.lc,self.err,profile),('observed_rv',self.rv,self.rv_err,True)]:
            if args[key] is None:continue
            residual=np.array(args[key]-model,copy=True);sigma=np.asarray(error)
            if do_profile:residual-=np.sum(residual/sigma**2)/np.sum(1/sigma**2)
            # Independent normalized Gaussian, with only fixed constants removed.
            result+=np.sum(norm.logpdf(residual,scale=sigma)-norm.logpdf(0.,scale=sigma))
        return result

    def test_single_and_batched_builders_match_absolute_gaussian_density(self):
        for photometry,rv in ((True,False),(False,True),(True,True)):
            for profile in (False,True):
                args=self.args(photometry,rv);expected=self.expected(args,profile)
                args['profile_lc_baseline']=profile
                for builder in (make_combined_chi2_vec,make_combined_chi2_vec_jax):
                    target=builder(**args,d=1.,gp=None,lensing_method='IntegratedWittMaoJax')
                    np.testing.assert_allclose(target(self.p[None]),[expected],atol=1e-9)
                batch={k:None if v is None else jnp.asarray(v)[None] for k,v in args.items() if k!='profile_lc_baseline'}
                batch['profile_lc_baseline']=profile
                for builder in (make_batched_combined_chi2_vec_jax,make_batched_numeric_combined_chi2_vec_jax):
                    extra=dict(d=jnp.ones(1),gp=None,lensing_method='IntegratedWittMaoJax') if builder is make_batched_combined_chi2_vec_jax else {}
                    target=builder(**batch,**extra)
                    np.testing.assert_allclose(target(self.p[None,None]),[[expected]],atol=1e-9)

    def test_masked_observation_and_gaussian_priors_are_not_rescaled(self):
        args=self.args(photometry=True,rv=False)
        args['lc_mask']=jnp.array([True,True,True,False])
        period,tau=physical_properties(parameter_row(self.p,1.,1.))[1:]
        args['tau_eff']=tau*1.01
        args['detected_periods']=jnp.array([period*1.02])
        expected=-.5*(1+4+.25)+norm.logpdf(float(period),loc=float(period*1.02),scale=.1)+norm.logpdf(float(tau),loc=float(tau*1.01),scale=float(tau*1.01)/100)
        for target in (combined_chi2,numeric_combined_chi2):
            extra=dict(d=1.,gp_model=None,lensing_method='IntegratedWittMaoJax') if target is combined_chi2 else {}
            np.testing.assert_allclose(target(params=self.p,**args,**extra),expected,atol=1e-9)

    def test_gp_density_is_unchanged_and_failure_uses_half_chi_squared(self):
        args=self.args(photometry=True,rv=False)
        args['lc_times']=SimpleNamespace(jd=self.t)
        epoch=.2;params=jnp.append(self.p,epoch)
        # Fixed generator outputs isolate weighting from the orbital model.
        flat=SimpleNamespace(flux_array=self.lc)
        for gp_value in (-7.,-jnp.inf):
            gp=SimpleNamespace(logprob_single=lambda:gp_value)
            with patch('selflensing.fitting.likelihood.LightcurveGenerator',return_value=flat),patch('selflensing.fitting.likelihood.GPLightcurveGenerator',return_value=SimpleNamespace(gp=gp)):
                actual=combined_chi2(params=params,**args,d=1.,gp_model=lambda *a:gp,lensing_method='IntegratedWittMaoJax',peak_time_prior_sigma=.5)
            expected=(float(gp_value) if np.isfinite(gp_value) else -.5*(1+4+.25+9))+norm.logpdf(epoch,scale=.5)
            np.testing.assert_allclose(actual,expected,atol=1e-9)
