"""Exact orbital coordinates based on Einstein strength and conjunction speed.

Coordinates: (log(RE0^2/Rstar^2), b_source, log(vt0/Rstar [day^-1]),
              eccentricity, periapsis_phase, [epoch_offset_day]).
No straight-track approximation is made when evaluating the light curve.
"""
import jax
import jax.numpy as jnp
import numpy as np
from astropy.constants import G,c,au,R_sun,M_sun
from astropy import units as u

MU=G.to_value(u.au**3/(u.M_sun*u.day**2))
C=c.to_value(u.au/u.day)
RS=R_sun.to_value(u.au)


def physical_to_pulse(position, companion_mass, radius_rsun):
    x=jnp.asarray(position,dtype=float)
    a,i,m,e,w=[x[...,k] for k in range(5)]
    radius=radius_rsun*RS
    f=1+e*jnp.cos(w)
    r=a*(1-e*e)/f
    strength=4*MU*m*r*jnp.sin(i)/(C*C*radius*radius)
    speed=jnp.sqrt(MU*(m+companion_mass)*f/r)/radius
    result=x.at[...,0].set(jnp.log(strength))
    result=result.at[...,1].set(r*jnp.cos(i)/radius)
    return result.at[...,2].set(jnp.log(speed))


def pulse_to_physical(position, companion_mass, radius_rsun):
    """Inverse with analytic log|d(a,i,M,e,w,t)/d(pulse coordinates)|.

    The scalar root is monotonic for positive mass. A safeguarded Newton
    iteration preserves the unique positive root; target bounds remain those
    of the original physical parameters.
    """
    z=jnp.asarray(position,dtype=float)
    strength,b,vs=jnp.exp(z[...,0]),z[...,1],jnp.exp(z[...,2])
    e,w=z[...,3],z[...,4]
    radius=radius_rsun*RS
    F=1+e*jnp.cos(w)
    A=(vs*radius)**2/(MU*F)
    K=4*MU/(C*C*radius*radius)
    required=strength/K
    # Ignoring b gives a lower estimate; r0+|b R| is an upper bracket.
    r0=(companion_mass+jnp.sqrt(companion_mass**2+4*A*required))/(2*A)
    B=jnp.abs(b*radius)
    lower=jnp.maximum(B,companion_mass/A)
    upper=r0+B
    def update(_,state):
        r,lo,hi=state
        rho=jnp.sqrt(jnp.maximum(r*r-B*B,jnp.finfo(z.dtype).tiny))
        mass=A*r-companion_mass
        residual=mass*rho-required
        lo=jnp.where(residual<0,r,lo);hi=jnp.where(residual>=0,r,hi)
        derivative=A*rho+mass*r/rho
        new=r-residual/derivative
        new=jnp.where((new>=lo)&(new<=hi)&jnp.isfinite(new),new,(lo+hi)/2)
        done=jnp.abs(residual)<=1e-14*jnp.maximum(required,1e-100)
        return jnp.where(done,r,new),lo,hi
    r,_,_=jax.lax.fori_loop(0,24,update,(upper,lower,upper))
    mass=A*r-companion_mass
    cosine=b*radius/r
    sine=jnp.sqrt(jnp.maximum(1-cosine*cosine,1e-300))
    inc=jnp.arccos(jnp.clip(cosine,0.,1.))
    a=r*F/(1-e*e)
    physical=z.at[...,0].set(a).at[...,1].set(inc).at[...,2].set(mass)
    factor=(1-e*e)/F
    jac=(jnp.log(2*radius)+jnp.log(sine)+jnp.log(mass)+jnp.log(mass+companion_mass)
         -jnp.log(factor)-jnp.log(mass+(mass+companion_mass)*sine*sine))
    residual=mass*jnp.sqrt(jnp.maximum(r*r-B*B,0.))-required
    valid=(b>=0)&(cosine>=0)&(cosine<1)&(mass>0)&(e>=0)&(e<1)&(F>0)&jnp.all(jnp.isfinite(physical),axis=-1)
    valid&=jnp.abs(residual)<=1e-9*jnp.maximum(required,1e-100)
    return physical,jnp.where(valid,jac,-jnp.inf)
