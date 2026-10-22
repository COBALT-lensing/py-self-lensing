"""Independent orbit, lensing, duration and public-API regression checks."""
import unittest
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from astropy import constants as c, units as u
from numpy.testing import assert_allclose
from scipy.optimize import brentq
from scipy.special import ellipe

from selflensing.orbits import orbital_phase, projected_position_m, einstein_radius_m
from selflensing.models import SelfLensingSystem
from selflensing.generators import SelfLensingLightcurveGenerator, RVGenerator
from selflensing.lightcurves import self_lensing_amplification, ellipsoidal_variations_amplification
from selflensing.fitting.numeric import physical_properties
from selflensing.porb import calc_a, calc_mass, calc_porb
from selflensing.gp.LensingGP import periodic_lensing_mean
from selflensing.gp import PriorsSamplers as ps
from selflensing.gp.PhysicalRelations import ellipsoidal_period_relation
from test_radius_convention import disc_magnification


def system(a=1., e=.8, peri=.7, i=np.pi/2-1e-5):
    return SelfLensingSystem(a=a*u.au,Macc=10*u.M_sun,Mcomp=1*u.M_sun,Rcomp=1*u.R_sun,
        Teff=5772*u.K,i=i*u.rad,e=e*u.one,periapsis_phase=peri*u.rad,d=1*u.kpc)


def row(model):
    return jnp.array([model.a.value,model.i.value,model.Macc.value,model.Mcomp.value,
                      model.Rcomp.value,model.Teff.value,model.e.value,model.periapsis_phase.value])


def time_at_longitude(model, phase):
    e=float(model.e.value); peri=float(model.periapsis_phase.value)
    def mean(anomaly):
        E=2*np.arctan2(np.sqrt(1-e)*np.sin(anomaly/2),np.sqrt(1+e)*np.cos(anomaly/2))
        return E-e*np.sin(E)
    return (mean(phase-peri)-mean(-peri))*float(model.porb.value)/(2*np.pi)


def position_at_time(model, time):
    e=float(model.e.value); w=float(model.periapsis_phase.value); i=float(model.i.value)
    E0=2*np.arctan2(np.sqrt(1-e)*np.sin(-w/2),np.sqrt(1+e)*np.cos(-w/2))
    M=(E0-e*np.sin(E0)+2*np.pi*time/float(model.porb.value))%(2*np.pi)
    E=brentq(lambda E:E-e*np.sin(E)-M,0.,2*np.pi,xtol=1e-14)
    x=(np.cos(E)-e)*np.sin(w)+np.sqrt(1-e*e)*np.sin(E)*np.cos(w)
    z=(np.cos(E)-e)*np.cos(w)-np.sqrt(1-e*e)*np.sin(E)*np.sin(w)
    return model.a.to(u.m).value*np.array([x,np.cos(i)*z,np.sin(i)*z])


class EccentricGeometryTest(unittest.TestCase):
    def test_high_e_phase_and_gradients_against_bracketed_kepler(self):
        for e in [0.,.6,.99,.999,.999999]:
            means=np.r_[np.geomspace(1e-15,.1,40),np.linspace(.1,np.pi,100),.2478570551688049]
            expected=[]; dmean=[]
            for M in means:
                E=brentq(lambda E:(1-e)*E+e*(E-np.sin(E))-M,0.,np.pi,
                         xtol=5e-324,rtol=4*np.finfo(float).eps)
                anomaly=2*np.arctan2(np.sqrt(1+e)*np.sin(E/2),np.sqrt(1-e)*np.cos(E/2))
                expected.append(anomaly)
                dmean.append(np.sqrt((1-e)*(1+e))/(1-e*np.cos(E))**2)
            def phase(M): return orbital_phase(M/(2*jnp.pi),1.,e,0.)
            actual=np.asarray(jax.jit(jax.vmap(phase))(jnp.asarray(means)))
            assert_allclose(np.angle(np.exp(1j*(actual-expected))),0.,atol=2e-9)
            grad=np.asarray(jax.jit(jax.vmap(jax.grad(phase)))(jnp.asarray(means)))
            assert_allclose(grad,dmean,rtol=3e-9,atol=1e-8)
        # The old phase snap erased this derivative at the reference epoch.
        model=system(e=.6,peri=.7)
        derivative=jax.grad(lambda t:orbital_phase(t,model.porb.value,.6,.7))(0.)
        self.assertGreater(float(derivative),0.)

    def test_projected_geometry_against_cartesian_kepler_orbit(self):
        model=system(e=.7,peri=1.1,i=1.2)
        times=np.linspace(-.4,.4,17)*float(model.porb.value)
        phase=orbital_phase(jnp.asarray(times),model.porb.value,.7,1.1)
        actual=np.asarray(projected_position_m(phase,model.a.to(u.m).value,1.2,.7,1.1)).T
        expected=np.array([position_at_time(model,t) for t in times])
        assert_allclose(actual,expected,rtol=2e-11,atol=.02)
        radii=np.asarray(model.rE(phase*u.rad).value)
        z=expected[:,2]
        assert_allclose(radii**2,4*c.G.value*10*c.M_sun.value*np.maximum(z,0)/c.c.value**2,
                        rtol=3e-11,atol=1.)

    def test_selection_reversals_and_no_stale_impact_cache(self):
        for a,w,expected in [(100.,0.,True),(30.,np.pi,False)]:
            model=system(a=a,e=.9,peri=w,i=np.deg2rad(89.99))
            radius=a*c.au.value*(1-.9**2)/(1+.9*np.cos(w))
            assert_allclose(model.b_source.value,radius*np.cos(np.deg2rad(89.99))/c.R_sun.value)
            self.assertEqual(bool(model.sl_possible),expected)
            self.assertEqual(bool(physical_properties(row(model))[0]),expected)
        model=system(e=0.,i=np.deg2rad(89.99))
        before=float(model.b.value)
        model.e=.9*u.one
        self.assertNotEqual(before,float(model.b.value))
        assert_allclose(model.b.value,model.b_source.value*model.Rcomp.to(u.m).value/model.rE().value)
        self.assertFalse(bool(system(i=0.).sl_possible))

    def test_dynamic_native_and_jax_profiles_against_independent_disc_integral(self):
        e,w=.8,.7; a=c.au.value; radius0=a*(1-e*e)/(1+e*np.cos(w))
        model=system(e=e,peri=w,i=np.arccos(.3*c.R_sun.value/radius0))
        inclination=float(model.i.value)
        times=[]; expected=[]
        for target in [.5,.75,1.5]:
            def projected(phi):
                r=a*(1-e*e)/(1+e*np.cos(phi-w))
                return r*np.sqrt(np.sin(phi)**2+np.cos(inclination)**2*np.cos(phi)**2)/c.R_sun.value
            phi=brentq(lambda phi:projected(phi)-target,0.,.2)
            t=time_at_longitude(model,phi); times.append(t)
            xyz=position_at_time(model,t)
            re=np.sqrt(4*c.G.value*10*c.M_sun.value*xyz[2]/c.c.value**2)
            expected.append(disc_magnification(np.hypot(*xyz[:2])/c.R_sun.value,re/c.R_sun.value))
        jax_values=self_lensing_amplification(jnp.array(times),row(model))
        native=SelfLensingLightcurveGenerator(model,times=jnp.array(times),peak_time=0.,lensing_method='IntegratedWittMao')
        assert_allclose(jax_values,expected,rtol=2e-5,atol=2e-5)
        assert_allclose(native.amplification,expected,rtol=2e-5,atol=2e-5)
        # Rear lens must produce exactly unity, in both backends, with finite gradients.
        rear=jnp.array([time_at_longitude(model,phi) for phi in [2.,np.pi,4.]])
        assert_allclose(self_lensing_amplification(rear,row(model)),1.,rtol=0,atol=0)
        native.times=rear
        assert_allclose(native.amplification,1.,rtol=0,atol=0)
        gradient=jax.jit(jax.grad(lambda p:jnp.sum(self_lensing_amplification(rear,p))))(row(model))
        self.assertTrue(np.isfinite(gradient).all())
        assert_allclose(gradient,0.,atol=0)

    def test_duration_matches_cartesian_local_track_and_period_inversion(self):
        model=system(e=.7,peri=.9,i=np.deg2rad(89.99))
        dt=1e-5
        point=position_at_time(model,0.)[:2]
        velocity=(position_at_time(model,dt)[:2]-position_at_time(model,-dt)[:2])/(2*dt*86400)
        closest2=np.dot(point,point)-np.dot(point,velocity)**2/np.dot(velocity,velocity)
        boundary=c.R_sun.value+float(model.rE().value)
        expected=2*np.sqrt(boundary**2-closest2)/np.linalg.norm(velocity)/86400
        assert_allclose(model.tau_eff.value,expected,rtol=2e-7)
        period=calc_porb(model.a.to(u.au),float(model.b.value),model.rE().to(u.m),
                          model.Rcomp.to(u.m),model.tau_eff.to(u.day),e=.7,periapsis_phase=.9)
        assert_allclose(period.value,model.porb.value,rtol=2e-12)
        mass=calc_mass(model.rE().to(u.m),model.a.to(u.m),b=float(model.b.value),e=.7,periapsis_phase=.9)
        assert_allclose(mass.to_value(u.M_sun),10.,rtol=2e-12)
        lo,hi=calc_a(model.rE().to(u.m),b=float(model.b.value),e=.7,periapsis_phase=.9)
        self.assertLess(lo.value,float(model.a.value)); self.assertGreater(hi.value,float(model.a.value))
        self.assertLess(float(model.tau_sl.value),float(model.tau_eff.value))

    def test_component_velocities_average_speed_and_rv_acceleration(self):
        model=system(e=.8,peri=.7)
        a=model.a.to(u.m).value; mu=c.G.value*11*c.M_sun.value
        for phi in [0.,.7,np.pi]:
            r=a*(1-.8**2)/(1+.8*np.cos(phi-.7))
            speed=np.sqrt(mu*(2/r-1/a))
            assert_allclose(model.v_tot(phi*u.rad).value,speed,rtol=2e-12)
            assert_allclose(model.v_comp(phi*u.rad).value,10/11*speed,rtol=2e-12)
            assert_allclose(model.v_acc(phi*u.rad).value,1/11*speed,rtol=2e-12)
        for ec in [0.,.8,.999999]:
            model.e=ec*u.one
            expected=4*a*ellipe(ec*ec)/(model.porb.value*86400)
            assert_allclose(model.v_avg.value,expected,rtol=2e-9)
        model.e=.7*u.one
        times=jnp.array([-.1,.1])*model.porb.value
        rv=RVGenerator(model,times=times,peak_time=0.)
        from selflensing.lightcurves import radial_velocity_m_per_s
        expected=jax.vmap(jax.grad(lambda t:radial_velocity_m_per_s(t,row(model))))(times)/86400
        assert_allclose(rv.rv_deriv.to(u.m/u.s**2).value,expected,rtol=2e-12)

    def test_equilibrium_tide_circular_limit_and_eccentric_axisymmetric_term(self):
        from selflensing.lightcurves import _EV_BETA
        for ec in [0.,.7]:
            model=system(e=ec,peri=.9,i=.8)
            times=np.array([-.1,.03,.2])*float(model.porb.value)
            expected=[]
            scale=_EV_BETA*10*(c.R_sun.value/c.au.value)**3
            for t in times:
                xyz=position_at_time(model,t); r=np.linalg.norm(xyz)
                expected.append(scale*(2/3*(1-3*(xyz[2]/r)**2)*(c.au.value/r)**3
                                      -(2/3-np.sin(.8)**2)))
            assert_allclose(ellipsoidal_variations_amplification(jnp.array(times),row(model)),expected,
                            rtol=1e-11,atol=1e-15)
        assert_allclose(ellipsoidal_period_relation(3.,0.),500.)
        assert_allclose(ellipsoidal_period_relation(3.,.7),1000.)

    def test_optional_gp_mean_has_physical_foreground_and_known_eccentricity(self):
        # Compare an aligned epoch to the analytic uniform source limit only via
        # the circular reference test above; here verify epoch symmetry/gating.
        pars=jnp.array([1.,10.,np.log10(100*86400),np.pi/2,0.,0.])
        times=jnp.array([0.,100*86400/2,100*86400])
        flux=periodic_lensing_mean(pars,times,eccentricity=.8,periapsis_phase=0.)
        self.assertGreater(float(flux[0]),1.)
        assert_allclose(flux[1],1.,rtol=0,atol=0)
        assert_allclose(flux[0],flux[2],rtol=2e-10)
        grad=jax.grad(lambda e:jnp.sum(periodic_lensing_mean(pars,times,eccentricity=e)))(.4)
        self.assertTrue(np.isfinite(grad))

    def test_gp_posterior_factories_forward_fixed_eccentric_geometry(self):
        from selflensing.gp.MCMCfitting import make_logposterior_fn
        theta = jnp.array([-4., -3., 1., 10., np.log10(100*86400), np.pi/2, 0., 0.])
        bounds = [(float(x)-.5, float(x)+.5) for x in theta]
        times = jnp.linspace(0., 50*86400, 5)
        values = []
        for ec in [0., .7]:
            posterior = make_logposterior_fn(kernel_type='G', nhyp=2,
                noise_measure=.01, param_bounds=bounds, lens_eccentricity=ec,
                lens_periapsis_phase=.7)
            value = posterior(theta, times, jnp.ones(5))
            self.assertTrue(np.isfinite(value))
            values.append(float(value))
        self.assertNotEqual(*values)

    def test_unknown_or_eccentric_gp_prior_does_not_force_two_per_orbit(self):
        first=jnp.array([-4.,50.,-3.]); second=first.at[0].set(-2.)
        for ec in [None,.8]:
            prior,_=ps.make_log_prior_physical_jit('E',eccentricity=ec)
            assert_allclose(prior(first,jnp.nan,5.,6.),prior(second,jnp.nan,5.,6.))
        prior,_=ps.make_log_prior_physical_jit('E',eccentricity=0.)
        self.assertNotEqual(float(prior(first,jnp.nan,5.,6.)),float(prior(second,jnp.nan,5.,6.)))


if __name__=='__main__': unittest.main()
