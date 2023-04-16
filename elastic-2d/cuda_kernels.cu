/*
    some basic funtions are copied from madagascar
    copyright@pyang, madagascar
*/
#define d_PI 3.141592653589793
/* cuda functions */
__global__ void cuda_set_sg(int *sxz, int sxbeg, int szbeg, int jsx, int jsz, int ns, int npml, int nnz)
/*< set the positions of sources and geophones in whole domain >*/
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < ns)
    {
        sxz[id] = nnz * (sxbeg + id * jsx + npml) + (szbeg + npml + id * jsz);
        // printf("sxz[%d] = %d .\n", id, sxz[id]); 
    }   
}

__global__ void cuda_ricker_wavelet(float *wlt, float fm, float dt, float nt)
/*< generate ricker wavelet with time deley >*/
{
    int it = threadIdx.x + blockDim.x * blockIdx.x;
    float tmp = d_PI * fm * fabsf(it * dt - 1.0 / fm);
    tmp *= tmp;
    if (it < nt) 
    {
        wlt[it] = (1.0 - 2.0 * tmp) * expf(-tmp);
        // printf("wlt[%d] = %f .\n", it, wlt[it]);
    }
}

__global__ void cuda_add_source(float *p, float *source, int *Sxz, int ns, bool add)
/*< add==true, add (inject) the source; add==false, subtract the source >*/
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < ns)
    {
        if (add)     p[Sxz[id]] += source[id];
        else         p[Sxz[id]] -= source[id];
    }
}

__global__ void cuda_forward_vel(float *vx, float *vxx, float *vxz, float *vz, float *vzx, float *vzz, float *vxp, float *vzp, float *vxs, float *vzs,
    float *txx, float *txz, float *tzz, float *d2x, float *d1z, float *rho, float *vp, float *vs, int nnz, int nnx, float dt, float dx, float dz, int npml)
/*< update vx using global device memory >*/
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int ix, iz, im;
    float diff1, diff2, diff3, diff4, diff5, diff6;

    ix = id / nnz;
    iz = id % nnz;
    int N = nnz * nnx;
    if (id >= mm && id < N - mm)
    {
        if (ix >= mm && ix < nnx - mm && iz >= npml && iz < nnz - mm)
        {
            diff1 = 0.;
            diff2 = 0.;
            diff3 = 0.;
            diff4 = 0.;
            diff5 = 0.;
            diff6 = 0.;

            for(im = 0; im < mm; im++)
            {
                diff1 += c[im] * (txx[id + (im + 1) * nnz] - txx[id - im * nnz]);
                diff2 += c[im] * (txz[id + im] - txz[id - im - 1]);
                diff3 += c[im] * (txz[id + im * nnz] - txz[id - (im + 1) * nnz]);
                diff4 += c[im] * (tzz[id + im + 1] - tzz[id - im]);
                diff5 += c[im] * ((txx[id + (im + 1) * nnz] - txx[id - im * nnz]) + (tzz[id + (im + 1) * nnz] - tzz[id - im * nnz]));
                diff6 += c[im] * ((tzz[id + im + 1] - tzz[id - im]) + (txx[id + im + 1] - txx[id - im]));
            }
            if (iz == npml)
            {
                diff1 *= (dt / (0.5 * rho[id] * dx));
                diff2 *= (dt / (0.5 * rho[id] * dz));
            }else
            {
                diff1 *= (dt / (rho[id] * dx));
                diff2 *= (dt / (rho[id] * dz));
            }
            diff3 *= (dt / (rho[id] * dx));
            diff4 *= (dt / (rho[id] * dz));
            diff5 *= dt * vp[id] * vp[id] / (rho[id] * dx * 2 * (vp[id] * vp[id] - vs[id] * vs[id]));
            diff6 *= dt * vp[id] * vp[id] / (rho[id] * dz * 2 * (vp[id] * vp[id] - vs[id] * vs[id]));

            if (ix >= npml && ix < nnx - npml && iz >= npml && iz < nnz - npml)
            {
                vx[id] += (diff1 + diff2);
                vz[id] += (diff3 + diff4); 
            }else 
            {
                vxx[id] = ((1 + 0.5 * dt * d2x[ix]) * vxx[id] + diff1) / (1 - 0.5 * dt * d2x[ix]);
                vxz[id] = ((1 + 0.5 * dt * d1z[iz]) * vxz[id] + diff2) / (1 - 0.5 * dt * d1z[iz]);
                vx[id] = vxx[id] + vxz[id];

                vzx[id] = ((1 + 0.5 * dt * d2x[ix]) * vzx[id] + diff3) / (1 - 0.5 * dt * d2x[ix]);
                vzz[id] = ((1 + 0.5 * dt * d1z[iz]) * vzz[id] + diff4) / (1 - 0.5 * dt * d1z[iz]);
                vz[id] = vzx[id] + vzz[id];
            }

            vxp[id] = ((1 + 0.5 * dt * d2x[ix]) * vxp[id] + diff5) / (1 - 0.5 * dt * d2x[ix]); 
            vzp[id] = ((1 + 0.5 * dt * d1z[iz]) * vzp[id] + diff6) / (1 - 0.5 * dt * d1z[iz]);
            vxs[id] = vx[id] - vxp[id];
            vzs[id] = vz[id] - vzp[id];
        }
    }
}

__global__ void cuda_forward_stress(float *txx, float *txxx, float *txxz, float *tzz, float *tzzx, float *tzzz, float *txz, float *txzx, float *txzz, 
    float *vx, float *vz, float *d2x, float *d1z, float *rho, float *vp, float *vs, int nnz, int nnx, float dt, float dx, float dz, int npml)
/*< update txx using global device memory>*/
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int ix, iz, im;
    float diff1, diff2, diff3, diff4, diff5, diff6;

    ix = id / nnz;
    iz = id % nnz;
    int N = nnz * nnx;
    if (id >= mm && id < N -mm)
    {
        if (ix >= mm && ix < nnx - mm && iz >= npml && iz < nnz - mm)
        {
            diff1 = 0.;
            diff2 = 0.;
            diff3 = 0.;
            diff4 = 0.;
            diff5 = 0.;
            diff6 = 0.;
            for(im = 0; im < mm; im++)
            {
                diff1 += c[im] * (vx[id + im * nnz] - vx[id - (im + 1) * nnz]);
                diff2 += c[im] * (vz[id + im] - vz[id - im - 1]);
                diff3 += c[im] * (vx[id + im * nnz] - vx[id - (im + 1) * nnz]);
                diff4 += c[im] * (vz[id + im] - vz[id - im - 1]);
                diff5 += c[im] * (vz[id + (im + 1) * nnz] - vz[id - im * nnz]);
                diff6 += c[im] * (vx[id + im + 1] - vx[id - im]);
            } 
            if (iz == npml)
            {
                tzz[id] = 0; txz[id] = 0;
                diff1 *= (2 * rho[id] * vs[id] * vs[id] * dt / dx);
                diff2 = 0;
                diff3 = 0;
                diff4 *= (2 * rho[id] * vs[id] * vs[id] * dt / dz);
            }else{
                diff1 *= (rho[id] * vp[id] * vp[id] * dt / dx);
                diff2 *= (rho[id] * (vp[id] * vp[id] - 2 * vs[id] * vs [id]) * dt / dz);
                diff3 *= (rho[id] * (vp[id] * vp[id] - 2 * vs[id] * vs [id]) * dt / dx);
                diff4 *= (rho[id] * vp[id] * vp[id] * dt / dz);
            }
                diff5 *= (rho[id] * vs[id] * vs[id] * dt / dx);
                diff6 *= (rho[id] * vs[id] * vs[id] * dt / dz);

            if (ix >= npml && ix < nnx - npml && iz >= npml && iz < nnz - npml)
            {
                txx[id] += (diff1 + diff2);
                tzz[id] += (diff3 + diff4);
                txz[id] += (diff5 + diff6);
            }else 
            {
                txxx[id] = ((1 + 0.5 * dt * d2x[ix]) * txxx[id] + diff1) / (1 - 0.5 * dt * d2x[ix]);
                txxz[id] = ((1 + 0.5 * dt * d1z[iz]) * txxz[id] + diff2) / (1 - 0.5 * dt * d1z[iz]);
                txx[id] = txxx[id] + txxz[id];

                tzzx[id] = ((1 + 0.5 * dt * d2x[ix]) * tzzx[id] + diff3) / (1 - 0.5 * dt * d2x[ix]);
                tzzz[id] = ((1 + 0.5 * dt * d1z[iz]) * tzzz[id] + diff4) / (1 - 0.5 * dt * d1z[iz]);
                tzz[id] = tzzx[id] + tzzz[id];

                txzx[id] = ((1 + 0.5 * dt * d2x[ix]) * txzx[id] + diff5) / (1 - 0.5 * dt * d2x[ix]);
                txzz[id] = ((1 + 0.5 * dt * d1z[iz]) * txzz[id] + diff6) / (1 - 0.5 * dt * d1z[iz]);
                txz[id] = txzx[id] + txzz[id];
            }
   

        }
    }

}

__global__ void cuda_backward_vel(float *vx, float *vz, float *txx, float *tzz, float *txz, float *vxp, float *vxs, float *vzp, float *vzs, 
    float *d2x, float *d1z, float *rho, float *vp, float *vs, int nnz, int nnx, float dt, float dx, float dz)
/* update vx, vz for receivers' wavefield && decouple wavefield */
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int ix, iz, im;
    float diff1, diff2, diff3, diff4, diff5, diff6;
    ix = id / nnz;
    iz = id % nnz;
    int N = nnz * nnx;
            
    // printf("iz = %d\n", ix);
             
    if (id >= mm && id < N - mm)
    {
        if (ix >= mm && ix < nnx - mm && iz >= mm && iz < nnz - mm)
        {
            diff1 = 0.;
            diff2 = 0.;
            diff3 = 0.;
            diff4 = 0.;
            diff5 = 0.;
            diff6 = 0.;
            for (im = 0; im < mm; im++)
            {
                diff1 += c[im] * (txx[id + im * nnz] - txx[id - (im + 1) * nnz]);
                diff2 += c[im] * (txz[id + im] - txz[id - (im + 1)]);
                diff3 += c[im] * (txz[id + (im + 1) * nnz] - txz[id - im * nnz]);
                diff4 += c[im] * (tzz[id + (im + 1)] - tzz[id - im]);
                diff5 += c[im] * ((txx[id + im * nnz] - txx[id - (im + 1) * nnz]) + (tzz[id + im * nnz] - tzz[id - (im + 1) * nnz]));
                diff6 += c[im] * ((tzz[id + im + 1] - tzz[id - im]) + (txx[id + im + 1] - txx[id - im]));
            }
            diff1 *= (dt / (rho[id] * dx));
            diff2 *= (dt / (rho[id] * dz));
            diff3 *= (dt / (rho[id] * dx));
            diff4 *= (dt / (rho[id] * dz));
            diff5 *= dt * vp[id] * vp[id] / (rho[id] * dx * 2 * (vp[id] * vp[id] - vs[id] * vs[id]));
            diff6 *= dt * vp[id] * vp[id] / (rho[id] * dz * 2 * (vp[id] * vp[id] - vs[id] * vs[id]));
            vx[id] -= (diff1 + diff2);
            vz[id] -= (diff3 + diff4);

            vxp[id] -= diff5; 
            vzp[id] -= diff6; 
            vxs[id] = vx[id] - vxp[id];
            vzs[id] = vz[id] - vzp[id];
        }
    }
}

__global__ void cuda_backward_stress(float *txx, float *tzz, float *txz, float *vx, float *vz,
    float *d2x, float *d1z, float *rho, float *vp, float *vs, int nnz, int nnx, float dt, float dx, float dz)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int ix, iz, im;
    float diff1, diff2, diff3, diff4, diff5, diff6;
    ix = id / nnz;
    iz = id % nnz;
    int N = nnz * nnx;
           
    if (id >= mm && id < N - mm)
    {
        if (ix >= mm && ix < nnx - mm && iz >= mm && iz < nnz - mm)
        {
            diff1 = 0.;
            diff2 = 0.;
            diff3 = 0.;
            diff4 = 0.;
            diff5 = 0.;
            diff6 = 0.;
            for (im = 0; im < mm; im++)
            {
                diff1 += c[im] * (vx[id + (im + 1) * nnz] - vx[id - im * nnz]);
                diff2 += c[im] * (vz[id + im] - vz[id - im - 1]);
                diff3 += c[im] * (vx[id + (im + 1) * nnz] - vx[id - im * nnz]);
                diff4 += c[im] * (vz[id + im] - vz[id - im - 1]);
                diff5 += c[im] * (vz[id + im * nnz] - vz[id - (im + 1) * nnz]);
                diff6 += c[im] * (vx[id + im + 1] - vx[id - im]);
            }
            diff1 *= (rho[id] * vp[id] * vp[id] * dt / dx);
            diff2 *= (rho[id] * (vp[id] * vp[id] - 2 * vs[id] * vs[id]) * dt / dz);
            diff3 *= (rho[id] * (vp[id] * vp[id] - 2 * vs[id] * vs[id]) * dt / dx);
            diff4 *= (rho[id] * vp[id] * vp[id] * dt / dz);
            diff5 *= (rho[id] * vs[id] * vs[id] * dt / dx);
            diff6 *= (rho[id] * vs[id] * vs[id] * dt / dz);
            txx[id] -= (diff1 + diff2);
            tzz[id] -= (diff3 + diff4);
            txz[id] -= (diff5 + diff6);

        }
    }
}

__global__ void cuda_record(float *record, float *p, int *Gxz, int ng)
/* record the seismogram at time kt */
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < ng)
    {
        record[id] = p[Gxz[id]];
    }
}

__global__ void cuda_mute(float *seis_kt, int gzbeg, int szbeg, int gxbeg, int sxc, int jgx, int kt, int ntd, float vmute, float dt, float dz, float dx, int ng)
/* mute direct wave */
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	float a = dx * abs(gxbeg + id * jgx - sxc);
	float b = dz * (gzbeg - szbeg);
	float t0 = sqrtf(a * a + b * b) / vmute;
	int ktt = int(t0 / dt) + ntd;// ntd is manually added to obtain the best muting effect.
	if (id < ng && kt < ktt)
    {
        seis_kt[id] = 0.0;
    } 
}

__global__ void cuda_boundary_rw(float *boundary, float *p, int npml, int nnx, int nnz, int it, bool save)
/* read or save boundary wavefield for effect boundary saving strategy */
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int ix, iz, nx, nz;
    ix = id / mm;
    iz = id % mm;
    nx = nnx - 2 * npml;
    nz = nnz - 2 * npml;

    if (save)
    {
        if (id < mm * nx) /* up boundary */
        {
            boundary[it * (2 * mm * (nx + nz)) + id] = p[nnz * (npml + ix) + iz + npml - mm];
        }
        if (id >= mm * nx && id < 2 * mm * nx) /* bottom boundary */
        {
            ix -= nx;
            boundary[it * (2 * mm * (nx + nz)) + id] = p[nnz * (npml + ix) + iz + npml + nz];
        }
        if (id >= 2 * mm * nx && id < 2 * mm * nx + mm * nz) /* left boundary */
        {
            ix -= 2 * nx;
            boundary[it * (2 * mm * (nx + nz)) + id] = p[nnz * (npml - mm + iz) + ix + npml];
        }
        if (id >= 2 * mm * nx + mm * nz && id < 2 * mm * (nx + nz)) /* right boundary */
        {
            ix -= (2 * nx + nz);
            boundary[it * (2 * mm * (nx + nz)) + id] = p[nnz * (npml + nx + iz) + ix + npml];
        }
    }
    else
    {
        if (id < mm * nx) /* up boundary */
        {
            p[nnz * (npml + ix) + iz + npml - mm] = boundary[it * (2 * mm * (nx + nz)) + id];
        }
        if (id >= mm * nx && id < 2 * mm * nx) /* bottom boundary */
        {
            ix -= nx;
            p[nnz * (npml + ix) + iz + npml + nz] = boundary[it * (2 * mm * (nx + nz)) + id];
        }
        if (id >= 2 * mm * nx && id < 2 * mm * nx + mm * nz) /* left boundary */
        {
            ix -= 2 * nx;
            p[nnz * (npml - mm + iz) + ix + npml] = boundary[it * (2 * mm * (nx + nz)) + id];
        }
        if (id >= 2 * mm * nx + mm * nz && id < 2 * mm * (nx + nz)) /* right boundary */
        {
            ix -= (2 * nx + nz);
            p[nnz * (npml + nx + iz) + ix + npml] = boundary[it * (2 * mm * (nx + nz)) + id];
        }
    }


}