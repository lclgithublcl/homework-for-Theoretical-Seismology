#define d_PI (4*atan(1.0))
/* cuda functions */
__global__ void cuda_set_sg(int *sxz, int sxbeg, int sybeg, int szbeg, int jsx, int jsy, int jsz, int ns, int npml, int nnz, int nnx)
/*< set the positions of sources and geophones in whole domain >*/
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < ns)
    {
        sxz[id] = nnz * nnx * (sybeg + id * jsy + npml) + nnz * (sxbeg + id * jsx + npml) + (szbeg + npml + id * jsz);
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

__global__ void cuda_forward_vel(float *vx, float *vxx, float *vxy, float *vxz, float *vy, float *vyx, float *vyy, float *vyz, 
float *vz, float *vzx, float * vzy, float *vzz, float *txx, float *tyy, float *tzz, float *txy, float *tyz, float *txz, 
float *d1z, float *d2x, float *d3y, float *vp, float *vs, float *rho, int nnz, int nnx, int nny, float dz, float dx, float dy, float dt, int npml)
/*< update velocity using global device memory >*/
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int ix = threadIdx.y + blockIdx.y * blockDim.y;
    float diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8, diff9;
    int id, iy, im;

    for (iy = 0; iy < nny; iy ++)
    {
        id = iz + ix * nnz + iy *nnz*nnx;
        if (id >= mm && id < nnx*nny*nnz - mm)
        {
            diff1 = 0.;  diff2 = 0.;  diff3 = 0.;
            diff4 = 0.;  diff5 = 0.;  diff6 = 0.;
            diff7 = 0.;  diff8 = 0.;  diff9 = 0.;
            if (ix >= mm && ix < nnx-mm && iy >= mm && iy < nny-mm && iz >= npml && iz < nnz-mm)
            {
                for (im = 0; im < mm; im ++)
                {
                    diff1 += c[im] * (txx[id + (im + 1) * nnz] - txx[id - im * nnz]);
                    diff2 += c[im] * (txy[id + im * nnz*nnx] - txy[id - (im + 1) * nnz*nnx]);
                    diff3 += c[im] * (txz[id + im] - txz[id - (im + 1)]);
                    diff4 += c[im] * (txy[id + im * nnz] - txy[id - (im + 1) * nnz]);
                    diff5 += c[im] * (tyy[id + (im + 1) *nnz*nnx] - tyy[id - im * nnz*nnx]);
                    diff6 += c[im] * (tyz[id + im] - tyz[id - (im + 1)]);
                    diff7 += c[im] * (txz[id + im * nnz] - txz[id - (im + 1) * nnz]);
                    diff8 += c[im] * (tyz[id + im * nnz*nnx] - tyz[id - (im + 1) * nnz*nnx]);
                    diff9 += c[im] * (tzz[id + im + 1] - tzz[id - im]);
                }
                if (iz == npml)
                {
                    diff1 *= dt / (0.5 * rho[id] * dx);
                    diff2 *= dt / (0.5 * rho[id] * dy);
                    diff3 *= dt / (0.5 * rho[id] * dz);
                    diff4 *= dt / (0.5 * rho[id] * dx);
                    diff5 *= dt / (0.5 * rho[id] * dy);
                    diff6 *= dt / (0.5 * rho[id] * dz);
                }else
                {
                    diff1 *= dt / (rho[id] * dx);
                    diff2 *= dt / (rho[id] * dy);
                    diff3 *= dt / (rho[id] * dz);
                    diff4 *= dt / (rho[id] * dx);   
                    diff5 *= dt / (rho[id] * dy);
                    diff6 *= dt / (rho[id] * dz);             
                }
                diff7 *= dt / (rho[id] * dx);
                diff8 *= dt / (rho[id] * dy);
                diff9 *= dt / (rho[id] * dz);
                if (iz >= npml && iz < nnz-npml && ix >= npml && ix < nnx-npml && iy >= npml && iy < nny-npml)
                {
                    vx[id] += diff1 + diff2 + diff3;
                    vy[id] += diff4 + diff5 + diff6;
                    vz[id] += diff7 + diff8 + diff9;
                }else
                {
                    vxx[id] = ((1 + 0.5 * dt * d2x[ix]) * vxx[id] + diff1) / (1 - 0.5 * dt * d2x[ix]);
                    vxy[id] = ((1 + 0.5 * dt * d3y[iy]) * vxy[id] + diff2) / (1 - 0.5 * dt * d3y[iy]);
                    vxz[id] = ((1 + 0.5 * dt * d1z[iz]) * vxz[id] + diff3) / (1 - 0.5 * dt * d1z[iz]);
                    vx[id] = vxx[id] + vxy[id] + vxz[id];

                    vyx[id] = ((1 + 0.5 * dt * d2x[ix]) * vyx[id] + diff4) / (1 - 0.5 * dt * d2x[ix]);
                    vyy[id] = ((1 + 0.5 * dt * d3y[iy]) * vyy[id] + diff5) / (1 - 0.5 * dt * d3y[iy]);
                    vyz[id] = ((1 + 0.5 * dt * d1z[iz]) * vyz[id] + diff6) / (1 - 0.5 * dt * d1z[iz]);
                    vy[id] = vyx[id] + vyy[id] + vyz[id];

                    vzx[id] = ((1 + 0.5 * dt * d2x[ix]) * vzx[id] + diff7) / (1 - 0.5 * dt * d2x[ix]);
                    vzy[id] = ((1 + 0.5 * dt * d3y[iy]) * vzy[id] + diff8) / (1 - 0.5 * dt * d3y[iy]);
                    vzz[id] = ((1 + 0.5 * dt * d1z[iz]) * vzz[id] + diff9) / (1 - 0.5 * dt * d1z[iz]);
                    vz[id] = vzx[id] + vzy[id] + vzz[id];
                }
            }
        }
    }
}

__global__ void cuda_forward_stress(
float *txx, float *txxx, float *txxy, float *txxz, float *tyy, float *tyyx, float *tyyy, float *tyyz, 
float *tzz, float *tzzx, float *tzzy, float *tzzz, float *txy, float *txyx, float *txyy, float *txyz, 
float *tyz, float *tyzx, float *tyzy, float *tyzz, float *txz, float *txzx, float *txzy, float *txzz, float *vx, float *vy, float *vz, 
float *d1z, float *d2x, float *d3y, float *vp, float *vs, float *rho, int nnz, int nnx, int nny, float dz, float dx, float dy, float dt, int npml)
/*< update stress using global device memory >*/
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int ix = threadIdx.y + blockIdx.y * blockDim.y;
    float diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8, diff9;
    float diff10, diff11, diff12, diff13, diff14, diff15;
    int id, iy, im;
    
    for (iy = 0; iy < nny; iy ++)
    {
        id = iz + ix * nnz + iy *nnz*nnx;
        if (id >= mm && id < nnx*nny*nnz - mm)
        {
            diff1 = 0.;  diff2 = 0.;  diff3 = 0.;
            diff4 = 0.;  diff5 = 0.;  diff6 = 0.;
            diff7 = 0.;  diff8 = 0.;  diff9 = 0.;
            diff10 = 0.; diff11 = 0.; diff12 = 0.;
            diff13 = 0.; diff14 = 0.; diff15 = 0.;
            if (ix >= mm && ix < nnx-mm && iy >= mm && iy < nny-mm && iz >= npml && iz < nnz-mm)
            {
                for (im = 0; im < mm; im ++)
                {
                    diff1 += c[im] * (vx[id + im * nnz] - vx[id - (im + 1) * nnz]); //txx
                    diff2 += c[im] * (vy[id + im * nnz*nnx] - vy[id - (im + 1) * nnz*nnx]);
                    diff3 += c[im] * (vz[id + im] - vz[id - (im + 1)]);
                    diff4 += c[im] * (vx[id + im * nnz] - vx[id - (im + 1) * nnz]); //tyy
                    diff5 += c[im] * (vy[id + im * nnz*nnx] - vy[id - (im + 1) * nnz*nnx]);
                    diff6 += c[im] * (vz[id + im] - vz[id - (im + 1)]);
                    diff7 += c[im] * (vx[id + im * nnz] - vx[id - (im + 1) * nnz]); //tzz
                    diff8 += c[im] * (vy[id + im * nnz*nnx] - vy[id - (im + 1) * nnz*nnx]);
                    diff9 += c[im] * (vz[id + im] - vz[id - (im + 1)]);
                    diff10 += c[im] * (vx[id + (im + 1) * nnz*nnx] - vx[id - im * nnz*nnx]); //txy
                    diff11 += c[im] * (vy[id + (im + 1) * nnz] - vy[id - im * nnz]);
                    diff12 += c[im] * (vy[id + (im + 1)] - vy[id - im]); //tyz
                    diff13 += c[im] * (vz[id + (im + 1) * nnz*nnx] - vz[id - im * nnz*nnx]);
                    diff14 += c[im] * (vx[id + (im + 1)] - vx[id - im]); //txz
                    diff15 += c[im] * (vz[id + (im + 1) * nnz] - vz[id - im * nnz]);
                }
                if (iz == npml)
                {
                    tzz[id] = 0;
                    diff1 *= (2 * rho[id] * vs[id] * vs[id] * dt / dx);
                    diff2 = 0;
                    diff3 = 0;
                    diff4 = 0;
                    diff5 *= (2 * rho[id] * vs[id] * vs[id] * dt / dy);
                    diff6 = 0;
                    diff7 = 0;
                    diff8 = 0;
                    diff9 *= (2 * rho[id] * vs[id] * vs[id] * dt / dz);;
                }else
                {
                    diff1 *= (rho[id] * vp[id] * vp[id] * dt) / dx; //txx
                    diff2 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dy;
                    diff3 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dz;
                    diff4 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dx; //tyy
                    diff5 *= (rho[id] * vp[id] * vp[id] * dt) / dy;
                    diff6 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dz;
                    diff7 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dx; //tzz
                    diff8 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dy;
                    diff9 *= (rho[id] * vp[id] * vp[id] * dt) / dz;
                }
                diff10 *= (rho[id] * vs[id] * vs[id] * dt) / dy; //txy
                diff11 *= (rho[id] * vs[id] * vs[id] * dt) / dx;
                diff12 *= (rho[id] * vs[id] * vs[id] * dt) / dz; //tyz
                diff13 *= (rho[id] * vs[id] * vs[id] * dt) / dy;
                diff14 *= (rho[id] * vs[id] * vs[id] * dt) / dz; //txz
                diff15 *= (rho[id] * vs[id] * vs[id] * dt) / dx;
                if (iz >= npml && iz < nnz-npml && ix >= npml && ix < nnx-npml && iy >= npml && iy < nny-npml)
                {
                    txx[id] += diff1 + diff2 + diff3;
                    tyy[id] += diff4 + diff5 + diff6;
                    tzz[id] += diff7 + diff8 + diff9;
                    txy[id] += diff10 + diff11;
                    tyz[id] += diff12 + diff13;
                    txz[id] += diff14 + diff15;
                }else
                {
                    txxx[id] = ((1 + 0.5 * dt * d2x[ix]) * txxx[id] + diff1) / (1 - 0.5 * dt * d2x[ix]);
                    txxy[id] = ((1 + 0.5 * dt * d3y[iy]) * txxy[id] + diff2) / (1 - 0.5 * dt * d3y[iy]);
                    txxz[id] = ((1 + 0.5 * dt * d1z[iz]) * txxz[id] + diff3) / (1 - 0.5 * dt * d1z[iz]);
                    txx[id] = txxx[id] + txxy[id] + txxz[id];

                    tyyx[id] = ((1 + 0.5 * dt * d2x[ix]) * tyyx[id] + diff4) / (1 - 0.5 * dt * d2x[ix]);
                    tyyy[id] = ((1 + 0.5 * dt * d3y[iy]) * tyyy[id] + diff5) / (1 - 0.5 * dt * d3y[iy]);
                    tyyz[id] = ((1 + 0.5 * dt * d1z[iz]) * tyyz[id] + diff6) / (1 - 0.5 * dt * d1z[iz]);
                    tyy[id] = tyyx[id] + tyyy[id] + tyyz[id];

                    tzzx[id] = ((1 + 0.5 * dt * d2x[ix]) * tzzx[id] + diff7) / (1 - 0.5 * dt * d2x[ix]);
                    tzzy[id] = ((1 + 0.5 * dt * d3y[iy]) * tzzy[id] + diff8) / (1 - 0.5 * dt * d3y[iy]);
                    tzzz[id] = ((1 + 0.5 * dt * d1z[iz]) * tzzz[id] + diff9) / (1 - 0.5 * dt * d1z[iz]);
                    tzz[id] = tzzx[id] + tzzy[id] + tzzz[id];

                    txyx[id] = ((1 + 0.5 * dt * d2x[ix]) * txyx[id] + diff11) / (1 - 0.5 * dt * d2x[ix]);
                    txyy[id] = ((1 + 0.5 * dt * d3y[iy]) * txyy[id] + diff10) / (1 - 0.5 * dt * d3y[iy]);
                    txy[id] = txyx[id] + txyy[id];

                    tyzy[id] = ((1 + 0.5 * dt * d3y[iy]) * tyzy[id] + diff13) / (1 - 0.5 * dt * d3y[iy]);
                    tyzz[id] = ((1 + 0.5 * dt * d1z[iz]) * tyzz[id] + diff12) / (1 - 0.5 * dt * d1z[iz]);
                    tyz[id] = tyzy[id] + tyzz[id];

                    txzx[id] = ((1 + 0.5 * dt * d2x[ix]) * txzx[id] + diff15) / (1 - 0.5 * dt * d2x[ix]);
                    txzz[id] = ((1 + 0.5 * dt * d1z[iz]) * txzz[id] + diff14) / (1 - 0.5 * dt * d1z[iz]);
                    txz[id] = txzx[id] + txzz[id];
                }
            }
        }
    }
}

__global__ void cuda_forward_vel_pml(float *vx, float *vxx, float *vxy, float *vxz, float *vy, float *vyx, float *vyy, float *vyz, 
float *vz, float *vzx, float * vzy, float *vzz, float *txx, float *tyy, float *tzz, float *txy, float *tyz, float *txz, 
float *d1z, float *d2x, float *d3y, float *vp, float *vs, float *rho, int nnz, int nnx, int nny, float dz, float dx, float dy, float dt, int npml)
/*< update velocity using global device memory >*/
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int ix = threadIdx.y + blockIdx.y * blockDim.y;
    float diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8, diff9;
    int id, iy, im;

    for (iy = 0; iy < nny; iy ++)
    {
        id = iz + ix * nnz + iy *nnz*nnx;
        if (id >= mm && id < nnx*nny*nnz - mm)
        {
            diff1 = 0.;  diff2 = 0.;  diff3 = 0.;
            diff4 = 0.;  diff5 = 0.;  diff6 = 0.;
            diff7 = 0.;  diff8 = 0.;  diff9 = 0.;
            if (ix >= mm && ix < nnx-mm && iy >= mm && iy < nny-mm && iz >= npml && iz < nnz-mm)
            {
                for (im = 0; im < mm; im ++)
                {
                    diff1 += c[im] * (txx[id + (im + 1) * nnz] - txx[id - im * nnz]);
                    diff2 += c[im] * (txy[id + im * nnz*nnx] - txy[id - (im + 1) * nnz*nnx]);
                    diff3 += c[im] * (txz[id + im] - txz[id - (im + 1)]);
                    diff4 += c[im] * (txy[id + im * nnz] - txy[id - (im + 1) * nnz]);
                    diff5 += c[im] * (tyy[id + (im + 1) *nnz*nnx] - tyy[id - im * nnz*nnx]);
                    diff6 += c[im] * (tyz[id + im] - tyz[id - (im + 1)]);
                    diff7 += c[im] * (txz[id + im * nnz] - txz[id - (im + 1) * nnz]);
                    diff8 += c[im] * (tyz[id + im * nnz*nnx] - tyz[id - (im + 1) * nnz*nnx]);
                    diff9 += c[im] * (tzz[id + im + 1] - tzz[id - im]);
                }

                diff1 *= dt / (rho[id] * dx);
                diff2 *= dt / (rho[id] * dy);
                diff3 *= dt / (rho[id] * dz);
                diff4 *= dt / (rho[id] * dx);   
                diff5 *= dt / (rho[id] * dy);
                diff6 *= dt / (rho[id] * dz);             
                diff7 *= dt / (rho[id] * dx);
                diff8 *= dt / (rho[id] * dy);
                diff9 *= dt / (rho[id] * dz);
                if (iz >= npml && iz < nnz-npml && ix >= npml && ix < nnx-npml && iy >= npml && iy < nny-npml)
                {
                    vx[id] += diff1 + diff2 + diff3;
                    vy[id] += diff4 + diff5 + diff6;
                    vz[id] += diff7 + diff8 + diff9;
                }else
                {
                    vxx[id] = ((1 + 0.5 * dt * d2x[ix]) * vxx[id] + diff1) / (1 - 0.5 * dt * d2x[ix]);
                    vxy[id] = ((1 + 0.5 * dt * d3y[iy]) * vxy[id] + diff2) / (1 - 0.5 * dt * d3y[iy]);
                    vxz[id] = ((1 + 0.5 * dt * d1z[iz]) * vxz[id] + diff3) / (1 - 0.5 * dt * d1z[iz]);
                    vx[id] = vxx[id] + vxy[id] + vxz[id];

                    vyx[id] = ((1 + 0.5 * dt * d2x[ix]) * vyx[id] + diff4) / (1 - 0.5 * dt * d2x[ix]);
                    vyy[id] = ((1 + 0.5 * dt * d3y[iy]) * vyy[id] + diff5) / (1 - 0.5 * dt * d3y[iy]);
                    vyz[id] = ((1 + 0.5 * dt * d1z[iz]) * vyz[id] + diff6) / (1 - 0.5 * dt * d1z[iz]);
                    vy[id] = vyx[id] + vyy[id] + vyz[id];

                    vzx[id] = ((1 + 0.5 * dt * d2x[ix]) * vzx[id] + diff7) / (1 - 0.5 * dt * d2x[ix]);
                    vzy[id] = ((1 + 0.5 * dt * d3y[iy]) * vzy[id] + diff8) / (1 - 0.5 * dt * d3y[iy]);
                    vzz[id] = ((1 + 0.5 * dt * d1z[iz]) * vzz[id] + diff9) / (1 - 0.5 * dt * d1z[iz]);
                    vz[id] = vzx[id] + vzy[id] + vzz[id];
                }
            }
        }
    }
}

__global__ void cuda_forward_stress_pml(
float *txx, float *txxx, float *txxy, float *txxz, float *tyy, float *tyyx, float *tyyy, float *tyyz, 
float *tzz, float *tzzx, float *tzzy, float *tzzz, float *txy, float *txyx, float *txyy, float *txyz, 
float *tyz, float *tyzx, float *tyzy, float *tyzz, float *txz, float *txzx, float *txzy, float *txzz, float *vx, float *vy, float *vz, 
float *d1z, float *d2x, float *d3y, float *vp, float *vs, float *rho, int nnz, int nnx, int nny, float dz, float dx, float dy, float dt, int npml)
/*< update stress using global device memory >*/
{
    int iz = threadIdx.x + blockIdx.x * blockDim.x;
    int ix = threadIdx.y + blockIdx.y * blockDim.y;
    float diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8, diff9;
    float diff10, diff11, diff12, diff13, diff14, diff15;
    int id, iy, im;
    
    for (iy = 0; iy < nny; iy ++)
    {
        id = iz + ix * nnz + iy *nnz*nnx;
        if (id >= mm && id < nnx*nny*nnz - mm)
        {
            diff1 = 0.;  diff2 = 0.;  diff3 = 0.;
            diff4 = 0.;  diff5 = 0.;  diff6 = 0.;
            diff7 = 0.;  diff8 = 0.;  diff9 = 0.;
            diff10 = 0.; diff11 = 0.; diff12 = 0.;
            diff13 = 0.; diff14 = 0.; diff15 = 0.;
            if (ix >= mm && ix < nnx-mm && iy >= mm && iy < nny-mm && iz >= npml && iz < nnz-mm)
            {
                for (im = 0; im < mm; im ++)
                {
                    diff1 += c[im] * (vx[id + im * nnz] - vx[id - (im + 1) * nnz]); //txx
                    diff2 += c[im] * (vy[id + im * nnz*nnx] - vy[id - (im + 1) * nnz*nnx]);
                    diff3 += c[im] * (vz[id + im] - vz[id - (im + 1)]);
                    diff4 += c[im] * (vx[id + im * nnz] - vx[id - (im + 1) * nnz]); //tyy
                    diff5 += c[im] * (vy[id + im * nnz*nnx] - vy[id - (im + 1) * nnz*nnx]);
                    diff6 += c[im] * (vz[id + im] - vz[id - (im + 1)]);
                    diff7 += c[im] * (vx[id + im * nnz] - vx[id - (im + 1) * nnz]); //tzz
                    diff8 += c[im] * (vy[id + im * nnz*nnx] - vy[id - (im + 1) * nnz*nnx]);
                    diff9 += c[im] * (vz[id + im] - vz[id - (im + 1)]);
                    diff10 += c[im] * (vx[id + (im + 1) * nnz*nnx] - vx[id - im * nnz*nnx]); //txy
                    diff11 += c[im] * (vy[id + (im + 1) * nnz] - vy[id - im * nnz]);
                    diff12 += c[im] * (vy[id + (im + 1)] - vy[id - im]); //tyz
                    diff13 += c[im] * (vz[id + (im + 1) * nnz*nnx] - vz[id - im * nnz*nnx]);
                    diff14 += c[im] * (vx[id + (im + 1)] - vx[id - im]); //txz
                    diff15 += c[im] * (vz[id + (im + 1) * nnz] - vz[id - im * nnz]);
                }
            
                diff1 *= (rho[id] * vp[id] * vp[id] * dt) / dx; //txx
                diff2 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dy;
                diff3 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dz;
                diff4 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dx; //tyy
                diff5 *= (rho[id] * vp[id] * vp[id] * dt) / dy;
                diff6 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dz;
                diff7 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dx; //tzz
                diff8 *= rho[id] * (vp[id] * vp[id] - 2*vs[id] * vs[id]) * dt / dy;
                diff9 *= (rho[id] * vp[id] * vp[id] * dt) / dz;
                diff10 *= (rho[id] * vs[id] * vs[id] * dt) / dy; //txy
                diff11 *= (rho[id] * vs[id] * vs[id] * dt) / dx;
                diff12 *= (rho[id] * vs[id] * vs[id] * dt) / dz; //tyz
                diff13 *= (rho[id] * vs[id] * vs[id] * dt) / dy;
                diff14 *= (rho[id] * vs[id] * vs[id] * dt) / dz; //txz
                diff15 *= (rho[id] * vs[id] * vs[id] * dt) / dx;
                if (iz >= npml && iz < nnz-npml && ix >= npml && ix < nnx-npml && iy >= npml && iy < nny-npml)
                {
                    txx[id] += diff1 + diff2 + diff3;
                    tyy[id] += diff4 + diff5 + diff6;
                    tzz[id] += diff7 + diff8 + diff9;
                    txy[id] += diff10 + diff11;
                    tyz[id] += diff12 + diff13;
                    txz[id] += diff14 + diff15;
                }else
                {
                    txxx[id] = ((1 + 0.5 * dt * d2x[ix]) * txxx[id] + diff1) / (1 - 0.5 * dt * d2x[ix]);
                    txxy[id] = ((1 + 0.5 * dt * d3y[iy]) * txxy[id] + diff2) / (1 - 0.5 * dt * d3y[iy]);
                    txxz[id] = ((1 + 0.5 * dt * d1z[iz]) * txxz[id] + diff3) / (1 - 0.5 * dt * d1z[iz]);
                    txx[id] = txxx[id] + txxy[id] + txxz[id];

                    tyyx[id] = ((1 + 0.5 * dt * d2x[ix]) * tyyx[id] + diff4) / (1 - 0.5 * dt * d2x[ix]);
                    tyyy[id] = ((1 + 0.5 * dt * d3y[iy]) * tyyy[id] + diff5) / (1 - 0.5 * dt * d3y[iy]);
                    tyyz[id] = ((1 + 0.5 * dt * d1z[iz]) * tyyz[id] + diff6) / (1 - 0.5 * dt * d1z[iz]);
                    tyy[id] = tyyx[id] + tyyy[id] + tyyz[id];

                    tzzx[id] = ((1 + 0.5 * dt * d2x[ix]) * tzzx[id] + diff7) / (1 - 0.5 * dt * d2x[ix]);
                    tzzy[id] = ((1 + 0.5 * dt * d3y[iy]) * tzzy[id] + diff8) / (1 - 0.5 * dt * d3y[iy]);
                    tzzz[id] = ((1 + 0.5 * dt * d1z[iz]) * tzzz[id] + diff9) / (1 - 0.5 * dt * d1z[iz]);
                    tzz[id] = tzzx[id] + tzzy[id] + tzzz[id];

                    txyx[id] = ((1 + 0.5 * dt * d2x[ix]) * txyx[id] + diff11) / (1 - 0.5 * dt * d2x[ix]);
                    txyy[id] = ((1 + 0.5 * dt * d3y[iy]) * txyy[id] + diff10) / (1 - 0.5 * dt * d3y[iy]);
                    txy[id] = txyx[id] + txyy[id];

                    tyzy[id] = ((1 + 0.5 * dt * d3y[iy]) * tyzy[id] + diff13) / (1 - 0.5 * dt * d3y[iy]);
                    tyzz[id] = ((1 + 0.5 * dt * d1z[iz]) * tyzz[id] + diff12) / (1 - 0.5 * dt * d1z[iz]);
                    tyz[id] = tyzy[id] + tyzz[id];

                    txzx[id] = ((1 + 0.5 * dt * d2x[ix]) * txzx[id] + diff15) / (1 - 0.5 * dt * d2x[ix]);
                    txzz[id] = ((1 + 0.5 * dt * d1z[iz]) * txzz[id] + diff14) / (1 - 0.5 * dt * d1z[iz]);
                    txz[id] = txzx[id] + txzz[id];
                }
            }
        }
    }
}

__global__ void cuda_record(float *record, float *p, int nnx, int nny, int nnz, int nx, int ny, int nz, int npml, int it, int nt)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int ix = id % nx;
    int iy = id / nx;

    if (id < nx * ny)
    {
        record[it + nt * ix + nt*nx * iy] = p[npml + nnz * (ix + npml) + nnz*nnx * (iy + npml)];
    }
}

// __global__ void cuda_add_source(float *p, float *source, int *Sxz, int ns, bool add)
// /*< add==true, add (inject) the source; add==false, subtract the source >*/
// {
//     int id = threadIdx.x + blockDim.x * blockIdx.x;
//     if (id < ns)
//     {
//         if (add)     p[Sxz[id]] += source[id];
//         else         p[Sxz[id]] -= source[id];
//     }
// }

__global__ void cuda_add_mv(float *vx, float *vy, float *vz, float *source, int *Sxz, float *Mv, int ns, 
                            int nnx, int nny, int nnz, float dx, float dy, float dz)
/* Mv: 0 --> Mxx; 1 --> Mxy; 2 --> Mxz;
       3 --> Myx; 4 --> Myy; 5 --> Myz;
       6 --> Mzx; 7 --> Mzy; 8 --> Mzz; */
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    float vol = dx * dy * dz; 
    
    if (id < ns)
    {
        vx[Sxz[id]]                 += source[id] / (dx * vol);  //Mxx
        vx[Sxz[id] - nnz]           -= Mv[0] * source[id] / (dx * vol);
        vx[Sxz[id] - nnz + nnz*nnx] += Mv[1] * source[id] / (4 * dy * vol); //Mxy
        vx[Sxz[id]       + nnz*nnx] += Mv[1] * source[id] / (4 * dy * vol);
        vx[Sxz[id] - nnz - nnz*nnx] -= Mv[1] * source[id] / (4 * dy * vol);
        vx[Sxz[id]       - nnz*nnx] -= Mv[1] * source[id] / (4 * dy * vol);
        vx[Sxz[id] - nnz + 1]       += Mv[2] * source[id] / (4 * dz * vol); //Mxz
        vx[Sxz[id]       + 1]       += Mv[2] * source[id] / (4 * dz * vol);
        vx[Sxz[id] - nnz - 1]       -= Mv[2] * source[id] / (4 * dz * vol);
        vx[Sxz[id]       - 1]       -= Mv[2] * source[id] / (4 * dz * vol);
        vy[Sxz[id] + nnz - nnz*nnx] += Mv[3] * source[id] / (4 * dx * vol); //Myx = Mxy
        vy[Sxz[id] + nnz          ] += Mv[3] * source[id] / (4 * dx * vol);
        vy[Sxz[id] - nnz - nnz*nnx] -= Mv[3] * source[id] / (4 * dx * vol);
        vy[Sxz[id] - nnz          ] -= Mv[3] * source[id] / (4 * dx * vol);
        vy[Sxz[id]]                 += Mv[4] * source[id] / (dy * vol); //Myy
        vy[Sxz[id] - nnz*nnx]       -= Mv[4] * source[id] / (dy * vol);
        vy[Sxz[id] - nnz*nnx + 1]   += Mv[5] * source[id] / (4 * dz * vol); //Myz
        vy[Sxz[id]           + 1]   += Mv[5] * source[id] / (4 * dz * vol);
        vy[Sxz[id] - nnz*nnx - 1]   -= Mv[5] * source[id] / (4 * dz * vol);
        vy[Sxz[id]           - 1]   -= Mv[5] * source[id] / (4 * dz * vol);
        vz[Sxz[id] + nnz - 1]       += Mv[6] * source[id] / (4 * dx * vol); //Mzx = Mxz
        vz[Sxz[id] + nnz    ]       += Mv[6] * source[id] / (4 * dx * vol);
        vz[Sxz[id] - nnz - 1]       -= Mv[6] * source[id] / (4 * dx * vol);
        vz[Sxz[id] - nnz    ]       -= Mv[6] * source[id] / (4 * dx * vol);
        vz[Sxz[id] + nnz*nnx - 1]   += Mv[7] * source[id] / (4 * dy * vol); //Mzy = Myz
        vz[Sxz[id] + nnz*nnx    ]   += Mv[7] * source[id] / (4 * dy * vol);
        vz[Sxz[id] - nnz*nnx - 1]   -= Mv[7] * source[id] / (4 * dy * vol);
        vz[Sxz[id] - nnz*nnx    ]   -= Mv[7] * source[id] / (4 * dy * vol);
        vz[Sxz[id]]                 += Mv[8] * source[id] / (dz * vol); //Mzz
        vz[Sxz[id] - 1]             += Mv[8] * source[id] / (dz * vol);
    }
}

