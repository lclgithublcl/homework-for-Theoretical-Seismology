/*
    GPU-based finite difference on 3-D grid
    coded by Chunlong Li @ UCAS, 2023/3/30 
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define EPS __FLT_EPSILON__
#define PI (4*atan(1.0))
#define Block_Size1 16
#define Block_Size2 16
const int npml = 50; /* thickness of PML boundary */
const int mm = 5;

__constant__ float c[mm] =  {1.211243,-0.08972168,0.01384277,-0.00176566,0.0001186795}; /*mm==5*/

#include "cuda_elastic_kernals.cu"
static int nx, ny, nz, nnx, nny, nnz, N, NJ, ns, nt;
static int jsx, jsy, jsz, sxbeg, sybeg, szbeg;
static float dx, dy, dz, fm, dt;

/* variables on host */
float *seis, *vp, *vs, *rho, *v0;
float *d1z, *d2x, *d3y; /* pml coeff */
float *h_snopshot;

/* variables on device */
int    *d_Sxz, *d_Gxz; /* set source and geophone position */
float  *d_d1z, *d_d2x, *d_d3y;
float  *d_wlt, *d_vp,  *d_vs,  *d_rho; /* wavelet, vp,vs,density */
float  *d_vx,  *d_vxx, *d_vxy, *d_vxz;
float  *d_vy,  *d_vyx, *d_vyy, *d_vyz;
float  *d_vz,  *d_vzx, *d_vzy, *d_vzz;
float  *d_txx, *d_txxx,*d_txxy,*d_txxz;
float  *d_tyy, *d_tyyx,*d_tyyy,*d_tyyz;
float  *d_tzz, *d_tzzx,*d_tzzy,*d_tzzz;
float  *d_txy, *d_txyx,*d_txyy,*d_txyz;
float  *d_tyz, *d_tyzx,*d_tyzy,*d_tyzz;
float  *d_txz, *d_txzx,*d_txzy,*d_txzz; 
float  *d_obsvx, *d_obsvy, *d_obsvz;
// float  *d_vxp, *d_vzp,  *d_vxs,  *d_vzs;

void expand(float *b, float *a, int npml, int nnz, int nnx, int nny, int nz, int nx, int ny)
/*< expand domain of 'a' to 'b':a, size=nz*nx*ny  b, size=nnz*nnx*nny >*/
{
    int ix, iy, iz;

    for (iy = 0; iy < ny; iy ++)
        for (ix = 0; ix < nx; ix ++)
            for(iz = 0; iz < nz; iz ++)
            {
                b[npml + iz + (npml + ix)*nnz + (npml + iy)*nnz*nnx] = a[iz + ix * nz + iy *nz*nx];
            }
    for (iy = 0; iy < nny; iy ++)
        for (ix = 0; ix < nnx; ix ++)
        {
            for (iz = 0; iz < npml; iz ++) b[iz + ix*nnz + iy*nnz*nnx] = b[npml + ix*nnz + iy*nnz*nnx]; //top
            for (iz = nz + npml; iz < nnz; iz ++) b[iz + ix*nnz + iy*nnz*nnx] = b[npml+nz-1 + ix*nnz + iy*nnz*nnx]; //bottom
        }
    for (iy = 0; iy < nny; iy ++)
        for (iz = 0; iz < nnz; iz ++)
        {
            for (ix = 0; ix < npml; ix ++) b[iz + ix*nnz + iy*nnz*nnx] = b[iz + npml*nnz + iy*nnz*nnx]; //left
            for (ix = nx + npml; ix < nnx; ix ++) b[iz + ix*nnz + iy*nnz*nnx] = b[iz + (npml+nx-1)*nnz + iy*nnz*nnx]; //right
        }
    for (ix = 0; ix < nnx; ix ++)
        for (iz = 0; iz < nnz; iz ++)
        {
            for (iy = 0; iy < npml; iy ++) b[iz + ix*nnz + iy*nnz*nnx] = b[iz + ix*nnz + npml*nnz*nnx]; //front
            for (iy = ny + npml; iy < nny; iy ++) b[iz + ix*nnz + iy*nnz*nnx] = b[iz + ix*nnz + (npml+nz-1)*nnz*nnx]; //back
        }
}

void window(float *a, float *b, int npml, int nnz, int nnx, int nny, int nz, int nx, int ny)
/*< window domain of 'b' to 'a':a, size=nz*nx*ny  b, size=nnz*nnx*nny >*/
{
    int iz, ix, iy;

    for (iy = 0; iy < ny; iy ++)
        for (ix = 0; ix < nx; ix ++)
            for (iz = 0; iz < nz; iz ++)
            {
                a[iz + ix*nz + iy*nz*nx] = b[npml+iz + (npml+ix)*nnz + (npml+iy)*nnz*nnx];
            }
}

void read_File(char Fvel[], int num, float *v)
/*< read file and write to v>*/
{
    FILE *fp;

    fp = fopen(Fvel, "rb");
    if (fp == NULL) printf("file does not exit!\n");
    for (int i = 0; i < num; i ++)
    {
        fread(&v[i], 4L, 1, fp);
    }
    fclose(fp);
}

void write_File(char Fout[], int num, float *out)
/*< write out to file >*/
{
    FILE *fp;

    fp = fopen(Fout,"wb");
    if(fp == NULL) printf("file does not exit!\n");
    for(int i = 0; i < num; i++)
        fwrite(&out[i], 4L, 1, fp);
    fclose(fp);
}

void check_grid_sanity(int NJ, float *vel, float fm, float dz, float dx, float dt, int N, float *vmax)
/*< sanity check about stability condition and non-dispersion condition >*/
{
    int i;
    float C;
    if(NJ==2) C=1;
    else if (NJ==4)	 	C=0.857;
    else if (NJ==6)		C=0.8;
    else if (NJ==8) 	C=0.777;
    else if (NJ==10)	C=0.759;

    float maxvel=vel[0], minvel=vel[0];
    for(i=0; i<N; i++)	{
	if(vel[i]>maxvel) maxvel=vel[i];
	if(vel[i]<minvel) minvel=vel[i];
    }
    vmax[0] = maxvel;
    float tmp=dt*maxvel*sqrtf(1.0/(dx*dx)+1.0/(dz*dz)+1.0/(dy*dy));

    if (tmp>=C) printf("Stability condition not satisfied!\n");
    if ( 	((NJ==2) &&(fm>=minvel/(10.0*max(dx,dz))))||
		((NJ==4) &&(fm>=minvel/(5.0*max(dx,dz))))	)
	printf("Non-dispersion relation not satisfied!\n");
}

void check_gpu_error(const char *msg) 
/*< check GPU errors >*/
{
    cudaError_t err = cudaGetLastError ();
    if (cudaSuccess != err) { 
	printf ("Cuda error: %s: %s", msg, cudaGetErrorString (err)); 
	exit(0);   
    }
}

void pmlcoeff_init(float *d1z, float *d2x, float *d3y, float vmax)
/*< initialize PML abosorbing coefficients >*/
{
    int ix, iz, iy;
    float Rc = 1.e6;
    float x, y, z, L = npml * (fmax(dx, dy) > dz ? fmax(dx, dy) : dz);
    float d0 = -3. * vmax * logf(Rc) / (2. * L * L * L) ;

    for (iz = 0; iz < nnz; iz++)
    {
        z = 0.;
        if (iz >= 0 && iz < npml)                 z = (npml - iz) * dz;  
        else if (iz >= nnz - npml && iz < nnz)    z = (iz - (nnz - npml -1)) * dz;
        d1z[iz] = d0 * z * z;
    }
    for (ix = 0; ix < nnx; ix++)
    {
        x = 0.;
        if (ix >= 0 && ix < npml)                 x = (ix - npml) * dx; //distance to inner field
        else if (ix >= nnx - npml && ix < nnx)    x = (ix - (nnx - npml -1)) * dx;
        d2x[ix] = d0 * x * x;
    }
    for (iy = 0; iy < nny; iy++)
    {
        y = 0.;
        if (iy >= 0 && iy < npml)                 y = (iy - npml) * dy; //distance to inner field
        else if (iy >= nny - npml && iy < nny)    y = (iy - (nny - npml -1)) * dy;
        d3y[iy] = d0 * y * y;
    }
}

int main()
{
    int is, it;
    float vmax;
    float Mv[9] = { 1, 0, 0, 
                    0, 1, 0,
                    0, 0, 1};
    float *d_Mv;
    /*< set up I/O files >*/
    char Fvp[90] = {"./model/junyun3d_101_101_51_vp.dat"};   /* velocity model, unit=m/s */
    char Fvs[90] = {"./model/junyun3d_101_101_51_vs.dat"};   /* vs */
    char Frho[90] = {"./model/junyun3d_101_101_51_rho.dat"};  /* rho */
    // char Fvz[90]={"./output/layers_vz.dat"}; 
    char Fvx[90]={"./output/junyun3d_vx.dat"};
    char Fvy[90]={"./output/junyun3d_vy.dat"};
    char Fvz[90]={"./output/junyun3d_vz.dat"};
    // char Fvxp[90]={"./output/layers_vxp.dat"};
    // char Fvxs[90]={"./output/layers_vxs.dat"};
    char Fobsvx[90]={"./output/junyun3d_obs_vx.dat"};
    char Fobsvy[90]={"./output/junyun3d_obs_vy.dat"};
    char Fobsvz[90]={"./output/junyun3d_obs_vz.dat"};
    FILE *fpsnopvx, *fpsnopvy, *fpsnopvz;
    fpsnopvx = fopen(Fvx, "wb");
    fpsnopvy = fopen(Fvy, "wb");
    fpsnopvz = fopen(Fvz, "wb");
    /* set parameters for elastic forward */ 
        nz=51; nx=101; ny=101; dx = 250; dy = 250; dz = 250;
        fm=1;  /* dominant freq of ricker */
        dt=0.025; /* time interval */
    	nt=400; /* total modeling time steps */
        ns=1;    /* total shots */
        jsx=0; /* source x-axis  jump interval  */
        jsy=0; 	/* source y-axis  jump interval  */
        jsz=0; 	/* source z-axis  jump interval  */
        sxbeg=50; /* x-begining index of sources, starting from 0 */
        sybeg=50; /* y-begining index of sources, starting from 0 */
        szbeg=0; /* z-begining index of sources, starting from 0 */
        NJ=10;  /* order of finite difference, order=2,4,6,8,10 */

    nnz = 2*npml + nz;
    nnx = 2*npml + nx;
    nny = 2*npml + ny;
    N = nnz*nnx*nny;

    dim3 dimg, dimb;
    dimg.x = (nnz + Block_Size1 - 1) / Block_Size1;
    dimg.y = (nnx + Block_Size2 - 1) / Block_Size2;
    dimb.x = Block_Size1;
    dimb.y = Block_Size2;
    /* allocate memory for host */
    const int M = nnz * nnx * nny * sizeof(float);
    v0  = (float *)malloc(nz * nx * ny * sizeof(float));
    vp  = (float *)malloc(M);
    vs  = (float *)malloc(M);
    rho = (float *)malloc(M);
    d1z = (float *)malloc(nnz * sizeof(float));
    d2x = (float *)malloc(nnx * sizeof(float));
    d3y = (float *)malloc(nny * sizeof(float));
    h_snopshot = (float *)malloc(M);
    seis = (float *)malloc(nx * ny * nt * sizeof(float));
    /* allocate variable space on device */
    cudaMalloc(&d_Sxz, ns  * sizeof(int));
    cudaMalloc(&d_d1z, nnz * sizeof(float));
    cudaMalloc(&d_d2x, nnx * sizeof(float));
    cudaMalloc(&d_d3y, nny * sizeof(float));
    cudaMalloc(&d_wlt, nt  * sizeof(float));
    cudaMalloc(&d_Mv,  9  * sizeof(float));
    cudaMalloc(&d_vp,  M);
    cudaMalloc(&d_vs,  M);
    cudaMalloc(&d_rho, M);
    cudaMalloc(&d_vx,  M);  cudaMalloc(&d_vxx,  M);   cudaMalloc(&d_vxy,  M);   cudaMalloc(&d_vxz,  M);
    cudaMalloc(&d_vy,  M);  cudaMalloc(&d_vyx,  M);   cudaMalloc(&d_vyy,  M);   cudaMalloc(&d_vyz,  M);
    cudaMalloc(&d_vz,  M);  cudaMalloc(&d_vzx,  M);   cudaMalloc(&d_vzy,  M);   cudaMalloc(&d_vzz,  M);
    cudaMalloc(&d_txx, M);  cudaMalloc(&d_txxx, M);   cudaMalloc(&d_txxy, M);   cudaMalloc(&d_txxz, M);
    cudaMalloc(&d_tyy, M);  cudaMalloc(&d_tyyx, M);   cudaMalloc(&d_tyyy, M);   cudaMalloc(&d_tyyz, M);
    cudaMalloc(&d_tzz, M);  cudaMalloc(&d_tzzx, M);   cudaMalloc(&d_tzzy, M);   cudaMalloc(&d_tzzz, M);
    cudaMalloc(&d_txy, M);  cudaMalloc(&d_txyx, M);   cudaMalloc(&d_txyy, M);   cudaMalloc(&d_txyz, M);
    cudaMalloc(&d_tyz, M);  cudaMalloc(&d_tyzx, M);   cudaMalloc(&d_tyzy, M);   cudaMalloc(&d_tyzz, M);
    cudaMalloc(&d_txz, M);  cudaMalloc(&d_txzx, M);   cudaMalloc(&d_txzy, M);   cudaMalloc(&d_txzz, M);
    cudaMalloc(&d_obsvx, nx * nt * nt * sizeof(float));
    cudaMalloc(&d_obsvy, nx * nt * nt * sizeof(float));
    cudaMalloc(&d_obsvz, nx * nt * nt * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    read_File(Fvp, nz*nx*ny, v0);
    expand(vp, v0, npml, nnz, nnx, nny, nz, nx, ny);
    read_File(Fvs, nz*nx*ny, v0);
    expand(vs, v0, npml, nnz, nnx, nny, nz, nx, ny);
    read_File(Frho, nz*nx*ny, v0);
    expand(rho, v0, npml, nnz, nnx, nny, nz, nx, ny);
    check_grid_sanity(NJ, rho, fm, dz, dx, dt, N, &vmax);
    printf("vmax = %f . \n", vmax);
    pmlcoeff_init(d1z, d2x, d3y, vmax);
    cuda_ricker_wavelet<<<(nt+511)/512,512>>>(d_wlt, fm, dt, nt);
    /* set source and geophone position */
    if (!(sxbeg>=0 && szbeg>=0 && sybeg>=0 && sxbeg+(ns-1)*jsx<nx && sybeg+(ns-1)*jsy<ny && szbeg+(ns-1)*jsz<nz))	
    { printf("sources exceeds the computing zone!"); exit(1);}
    cuda_set_sg<<<(ns + 255) / 256, 256>>>(d_Sxz, sxbeg, sybeg, szbeg, jsx, jsy, jsz, ns, npml, nnz, nnx);
    // fwrite(vp, 4L, N, fpsnopvx);

    cudaMemcpy(d_d1z, d1z, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d2x, d2x, nnx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d3y, d3y, nny * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vp,  vp,  M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs,  vs,  M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, rho, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mv,  Mv,  9 * sizeof(float), cudaMemcpyHostToDevice);
    for (is = 0; is < ns; is++)
    {
        cudaEventRecord(start);
        cudaMemset(d_vx,  0, M);   cudaMemset(d_vxx,  0, M);   cudaMemset(d_vxy,  0, M);   cudaMemset(d_vxz,  0, M);
        cudaMemset(d_vy,  0, M);   cudaMemset(d_vyx,  0, M);   cudaMemset(d_vyy,  0, M);   cudaMemset(d_vyz,  0, M);
        cudaMemset(d_vz,  0, M);   cudaMemset(d_vzx,  0, M);   cudaMemset(d_vzy,  0, M);   cudaMemset(d_vzz,  0, M);
        cudaMemset(d_txx, 0, M);   cudaMemset(d_txxx, 0, M);   cudaMemset(d_txxy, 0, M);   cudaMemset(d_txxz, 0, M);
        cudaMemset(d_tyy, 0, M);   cudaMemset(d_tyyx, 0, M);   cudaMemset(d_tyyy, 0, M);   cudaMemset(d_tyyz, 0, M);
        cudaMemset(d_tzz, 0, M);   cudaMemset(d_tzzx, 0, M);   cudaMemset(d_tzzy, 0, M);   cudaMemset(d_tzzz, 0, M);
        cudaMemset(d_txy, 0, M);   cudaMemset(d_txyx, 0, M);   cudaMemset(d_txyy, 0, M);   cudaMemset(d_txyz, 0, M);
        cudaMemset(d_txz, 0, M);   cudaMemset(d_txzx, 0, M);   cudaMemset(d_txzy, 0, M);   cudaMemset(d_txzz, 0, M);
        cudaMemset(d_tyz, 0, M);   cudaMemset(d_tyzx, 0, M);   cudaMemset(d_tyzy, 0, M);   cudaMemset(d_tyzz, 0, M);

        for (it = 0; it < nt; it ++)
        {
            // cuda_add_source<<<1, 1>>>(d_vx, &d_wlt[it], &d_Sxz[is], ns, true);
            // cuda_add_source<<<1, 1>>>(d_vy, &d_wlt[it], &d_Sxz[is], ns, true);
            // cuda_add_source<<<1, 1>>>(d_vz, &d_wlt[it], &d_Sxz[is], ns, false);
            cuda_add_mv<<<1, 1>>>(d_vx, d_vy, d_vz, &d_wlt[it], d_Sxz, d_Mv, ns, nnx, nny, nnz, dx, dy, dz);
            cuda_forward_vel_pml<<<dimg, dimb>>>(d_vx, d_vxx, d_vxy, d_vxz, d_vy, d_vyx, d_vyy, d_vyz, d_vz, d_vzx, d_vzy, d_vzz, d_txx, d_tyy, d_tzz, d_txy, d_tyz, d_txz, 
                                                    d_d1z,  d_d2x, d_d3y, d_vp, d_vs, d_rho, nnz, nnx, nny, dz, dx, dy, dt, npml);
            cuda_forward_stress_pml<<<dimg, dimb>>>(d_txx, d_txxx, d_txxy, d_txxz, d_tyy, d_tyyx, d_tyyy, d_tyyz, d_tzz, d_tzzx, d_tzzy, d_tzzz, 
                                                    d_txy, d_txyx, d_txyy, d_txyz, d_tyz, d_tyzx, d_tyzy, d_tyzz, d_txz, d_txzx, d_txzy, d_txzz, d_vx, d_vy, d_vz, 
                                                    d_d1z,  d_d2x, d_d3y, d_vp, d_vs, d_rho, nnz, nnx, nny, dz, dx, dy, dt, npml);
            cuda_record<<<(nx * ny + 511) / 512, 512>>>(d_obsvx, d_vx, nnx, nny, nnz, nx, ny, nz, npml, it, nt);
            cuda_record<<<(nx * ny + 511) / 512, 512>>>(d_obsvy, d_vy, nnx, nny, nnz, nx, ny, nz, npml, it, nt);
            cuda_record<<<(nx * ny + 511) / 512, 512>>>(d_obsvz, d_vz, nnx, nny, nnz, nx, ny, nz, npml, it, nt);
            
            if ((it == 150) && (is == 0) && it != 0)
            {
                printf("it = %d \n", it);
                cudaMemcpy(h_snopshot, d_vx, M, cudaMemcpyDeviceToHost);
                window(v0, h_snopshot, npml, nnz, nnx, nny, nz, nx, ny);
                fwrite(v0, 4L, nz*nx*ny, fpsnopvx);

                cudaMemcpy(h_snopshot, d_vy, M, cudaMemcpyDeviceToHost);
                window(v0, h_snopshot, npml, nnz, nnx, nny, nz, nx, ny);
                fwrite(v0, 4L, nz*nx*ny, fpsnopvy);

                cudaMemcpy(h_snopshot, d_vz, M, cudaMemcpyDeviceToHost);
                window(v0, h_snopshot, npml, nnz, nnx, nny, nz, nx, ny);
                fwrite(v0, 4L, nz*nx*ny, fpsnopvz);
            }
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) 
            {
                printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }
        }
    }

    /*< output record seismogram >*/
    cudaMemcpy(seis, d_obsvx, nt * nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
    write_File(Fobsvx, nt * nx * ny, seis);
    cudaMemcpy(seis, d_obsvy, nt * nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
    write_File(Fobsvy, nt * nx * ny, seis);
    cudaMemcpy(seis, d_obsvz, nt * nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
    write_File(Fobsvz, nt * nx * ny, seis);              
                
    free(v0); free(vp); free(vs); free(rho); 
    free(d1z); free(d2x); free(d3y); free(h_snopshot);
    
    cudaFree(d_d1z);    cudaFree(d_d2x);    cudaFree(d_d3y);
    cudaFree(d_Gxz);    cudaFree(d_Sxz); 
    cudaFree(d_vp);     cudaFree(d_vs);     cudaFree(d_rho);
    cudaFree(d_obsvx);  cudaFree(d_obsvy);  cudaFree(d_obsvz);
    cudaFree(d_vx);     cudaFree(d_vxx);    cudaFree(d_vxy);    cudaFree(d_vxz); 
    cudaFree(d_vy);     cudaFree(d_vyx);    cudaFree(d_vyy);    cudaFree(d_vyz);
    cudaFree(d_vz);     cudaFree(d_vzx);    cudaFree(d_vzy);    cudaFree(d_vzz);
    cudaFree(d_txx);    cudaFree(d_txxx);   cudaFree(d_txxy);   cudaFree(d_txxz);
    cudaFree(d_tyy);    cudaFree(d_tyyx);   cudaFree(d_tyyy);   cudaFree(d_tyyz);
    cudaFree(d_tzz);    cudaFree(d_tzzx);   cudaFree(d_tzzy);   cudaFree(d_tzzz);
    cudaFree(d_txy);    cudaFree(d_txyx);   cudaFree(d_txyy);   cudaFree(d_txyz);
    cudaFree(d_tyz);    cudaFree(d_tyzx);   cudaFree(d_tyzy);   cudaFree(d_tyzz);
    cudaFree(d_txz);    cudaFree(d_txzx);   cudaFree(d_txzy);   cudaFree(d_txzz);



}