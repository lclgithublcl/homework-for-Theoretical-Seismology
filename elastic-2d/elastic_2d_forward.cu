/*
    2D elastic forward code using standard stagger grid finite-difference method
    && pml boundary condition with freesurface 
                code by Chunlong Li @UCAS 
                2023 / 4 / 16
*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<cuda_runtime.h>

#define EPS __FLT_EPSILON__
#define PI 3.141592653589793
#define Block_Size1 16
#define Block_Size2 16
const int npml = 50; /* thickness of PML boundary */
const int mm = 5;

__constant__ float c[mm] = {1.211243,-0.08972168,0.01384277,-0.00176566,0.0001186795}; /*mm==5*/

#include "cuda_kernels.cu"
static bool  csdgather; /* common shot gather or not */
static int   nz1,nx1,nz,nx,nnz,nnx,N,NJ,ns,ng,nt,ntd;
static int   jsx,jsz,jgx,jgz,sxbeg,szbeg,gxbeg,gzbeg;
static float fm,dt,dz,dx,vmute;

/* variables on host */
float *seis, *vp, *vs, *rho, *v0;
float *d1z, *d2x;  /* pml coeff */
/* variables on device */
int    *d_Sxz, *d_Gxz; /* set source and geophone position */
float  *d_d1z, *d_d2x;
float  *d_wlt, *d_obsvx, *d_obsvz, *d_vp,  *d_vs,  *d_rho; /* wavelet, seismograms, vp,vs,density */
float  *d_vx,  *d_vxx, *d_vxz;
float  *d_vz,  *d_vzx, *d_vzz;
float  *d_txx, *d_txxx,*d_txxz;
float  *d_tzz, *d_tzzx,*d_tzzz;
float  *d_txz, *d_txzx,*d_txzz; 
float  *d_vxp, *d_vzp,  *d_vxs,  *d_vzs;

float  *d_boundary_vx, *d_boundary_vz; /* boundary on device */
float  *h_snopshot; /* snopshot */
float  *ptr=NULL;

void expand(float *b, float *a, int npml, int nnz, int nnx, int nz, int nx)
/*< expand domain of 'a' to 'b':a, size=nz*nx  b, size=nnz*nnx >*/
{
    int iz,ix;
    for (ix = 0; ix < nx; ix++)
    {
        for (iz = 0; iz < nz; iz++)
        {
            b[(npml+ix)*nnz+(npml+iz)] = a[ix*nz+iz];
        }
    }
    for (ix = 0; ix < nnx; ix++)
    {
        for (iz = 0; iz < npml; iz++) b[ix*nnz+iz] = b[ix*nnz+npml]; //top
        for (iz = nz + npml; iz < nnz; iz++) b[ix*nnz+iz] = b[ix*nnz+npml+nz-1]; //bottom
    } 
    for (iz = 0; iz < nnz; iz++)
    {
        for (ix = 0; ix < npml; ix++) b[ix*nnz+iz] = b[npml*nnz+iz]; //left
        for (ix = nx + npml; ix < nnx; ix++) b[ix*nnz+iz] = b[(npml+nx-1)*nnz+iz]; //right
    }
}

void window(float *a, float *b, int npml, int nnz, int nnx, int nz, int nx)
/*< window domain of 'b' to 'a':a, size=nz*nx  b, size=nnz*nnx >*/
{
    int ix, iz;
    for (ix = 0; ix < nx; ix++)
    {
        for (iz = 0; iz < nz; iz++)
        {
            a[ix*nz+iz] = b[(npml+ix)*nnz+(npml+iz)];
        }
    }
}
void read_File(char Fvel[], int num, float *v)
/*< read file and write to v>*/
{
    FILE *fp;

    fp = fopen(Fvel,"rb");
    if(fp == NULL) printf("file does not exit!\n");
    for(int i = 0; i < num; i++)
        {
            fread(&v[i], 4L, 1, fp);
        }
}

void write_File(char Fout[], int num, float *out)
/*< write out to file >*/
{
    FILE *fp;

    fp = fopen(Fout,"wb");
    if(fp == NULL) printf("file does not exit!\n");
    for(int i = 0; i < num; i++)
        fwrite(&out[i], 4L, 1, fp);
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
    float tmp=dt*maxvel*sqrtf(1.0/(dx*dx)+1.0/(dz*dz));

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

void pmlcoeff_init(float *d1z, float *d2x, float vmax)
/*< initialize PML abosorbing coefficients >*/
{
    int ix, iz;
    float Rc = 1.e6;
    float x, z, L = npml * (dx > dz ? dx : dz);
    float d0 = -3. * vmax * logf(Rc) / (2. * L * L * L) ;

    for (ix = 0; ix < nnx; ix++)
    {
        x = 0.;
        if (ix >= 0 && ix < npml)                 x = (ix - npml) * dx; //distance to inner field
        else if (ix >= nnx - npml && ix < nnx)    x = (ix - (nnx - npml -1)) * dx;
        d2x[ix] = d0 * x * x;
    }
    for (iz = 0; iz < nnz; iz++)
    {
        z = 0.;
        if (iz >= 0 && iz < npml)                 z = (npml - iz) * dz;  
        else if (iz >= nnz - npml && iz < nnz)    z = (iz - (nnz - npml -1)) * dz;
        d1z[iz] = d0 * z * z;
    }
}

int main()
{
    int is, it, distx, distz;
    float vmax;
    /*< set up I/O files >*/
    char Fvp[90] = {"./model/junyun_vp_400_400.dat"};   /* velocity model, unit=m/s */
    char Fvs[90] = {"./model/junyun_vs_400_400.dat"};   /* vs */
    char Frho[90] = {"./model/junyun_rho_400_400.dat"};  /* rho */
    char Fvx[90]={"./output/junyun_vx.dat"};
    char Fvz[90]={"./output/junyun_vz.dat"};
    char Fvxp[90]={"./output/junyun_vxp.dat"};
    char Fvxs[90]={"./output/junyun_vxs.dat"};
    char Fobsvx[90]={"./output/junyun_obsvx.dat"};
    char Fobsvz[90]={"./output/junyun_obsvz.dat"};
    FILE *fpsnopvx, *fpsnopvz, *fpshotvx, *fpshotvz, *fpsnopvxp, *fpsnopvxs;
    fpsnopvx = fopen(Fvx, "wb");
    fpsnopvz = fopen(Fvz, "wb");
    fpsnopvxp = fopen(Fvxp, "wb");
    fpsnopvxs = fopen(Fvxs, "wb");
    fpshotvx = fopen(Fobsvx, "wb");
    fpshotvz = fopen(Fobsvz, "wb");
    /* set parameters for elastic forward */ 
        nz1=400;  nx1=400;  dz=10;  dx=10;
        fm=15;  /* dominant freq of ricker */
        dt=0.001; /* time interval */
    	nt=1000; /* total modeling time steps */
        ns=1;    /* total shots */
    	ng=400;	/* total receivers in each shot */
        jsx=50; /* source x-axis  jump interval  */
        jsz=0; 	/* source z-axis  jump interval  */
        jgx=1;  /* receiver x-axis jump interval */
        jgz=0;  /* receiver z-axis jump interval */
        sxbeg=200; /* x-begining index of sources, starting from 0 */
        szbeg=0; /* z-begining index of sources, starting from 0 */
        gxbeg=0; /* x-begining index of receivers, starting from 0 */
        gzbeg=5; /* z-begining index of receivers, starting from 0 */
        NJ=10;  /* order of finite difference, order=2,4,6,8,10 */
        csdgather=false; /* default, common shot-gather; if n, record at every point*/
        ntd=2.0/(fm*dt);/* number of deleyed time samples to mute */
        vmute=2400.0;      /* muting velocity to remove the low-freq artifacts, unit=m/s*/

	// nx=(int)((nx1+Block_Size1-1)/Block_Size1)*Block_Size1; // set nx to be times of Block_Size1 xiang shang qu zheng
	// nz=(int)((nz1+Block_Size2-1)/Block_Size2)*Block_Size2; // set nz to be times of Block_Size2 
    nnz = 2*npml+nz1;
    nnx = 2*npml+nx1;
	N=nnz*nnx;
    
    /* allocate memory for host */
    const int M = nnz * nnx * sizeof(float);
    v0  = (float *)malloc(nx1 * nz1 * sizeof(float));
    vp  = (float *)malloc(M);
    vs  = (float *)malloc(M);
    rho = (float *)malloc(M);
    d1z = (float *)malloc(nnz * sizeof(float));
    d2x = (float *)malloc(nnx * sizeof(float));
    h_snopshot = (float *)malloc(M);
    seis = (float *)malloc(ng * nt * sizeof(float));
    /* allocate variable space on device */
    cudaMalloc(&d_Sxz, ns * sizeof(int));
    cudaMalloc(&d_Gxz, ng * sizeof(int));
    cudaMalloc(&d_d1z, nnz * sizeof(float));
    cudaMalloc(&d_d2x, nnx * sizeof(float));
    cudaMalloc(&d_wlt, nt * sizeof(float));
    cudaMalloc(&d_obsvx, nt * ng * sizeof(float));
    cudaMalloc(&d_obsvz, nt * ng * sizeof(float));
    cudaMalloc(&d_boundary_vx, nt * 2 * mm * (nx1 + nz1) * sizeof(float));
    cudaMalloc(&d_boundary_vz, nt * 2 * mm * (nx1 + nz1) * sizeof(float));
    cudaMalloc(&d_vp,  M);
    cudaMalloc(&d_vs,  M);
    cudaMalloc(&d_rho, M);
    cudaMalloc(&d_vx,  M);  cudaMalloc(&d_vxx,  M);   cudaMalloc(&d_vxz,  M);
    cudaMalloc(&d_vz,  M);  cudaMalloc(&d_vzx,  M);   cudaMalloc(&d_vzz,  M);
    cudaMalloc(&d_txx, M);  cudaMalloc(&d_txxx, M);   cudaMalloc(&d_txxz, M);
    cudaMalloc(&d_tzz, M);  cudaMalloc(&d_tzzx, M);   cudaMalloc(&d_tzzz, M);
    cudaMalloc(&d_txz, M);  cudaMalloc(&d_txzx, M);   cudaMalloc(&d_txzz, M);
    cudaMalloc(&d_vxp, M);  cudaMalloc(&d_vxs,  M);   cudaMalloc(&d_vzp,  M);   cudaMalloc(&d_vzs,  M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    read_File(Fvp, nz1*nx1, v0);
    expand(vp, v0, npml, nnz, nnx, nz1, nx1);
    read_File(Fvs, nz1*nx1, v0);
    expand(vs, v0, npml, nnz, nnx, nz1, nx1);
    read_File(Frho, nz1*nx1, v0);
    expand(rho, v0, npml, nnz, nnx, nz1, nx1);
    check_grid_sanity(NJ, vp, fm, dz, dx, dt, N, &vmax);
    printf("vmax = %f . \n", vmax);
    pmlcoeff_init(d1z, d2x, vmax);
    cuda_ricker_wavelet<<<(nt+511)/512,512>>>(d_wlt, fm, dt, nt);
    /* set source and geophone position */
    if (!(sxbeg>=0 && szbeg>=0 && sxbeg+(ns-1)*jsx<nx1 && szbeg+(ns-1)*jsz<nz1))	
    { printf("sources exceeds the computing zone!"); exit(1);}
    cuda_set_sg<<<(ns + 255) / 256, 256>>>(d_Sxz, sxbeg, szbeg, jsx, jsz, ns, npml, nnz);
    if (!(gxbeg>=0 && gzbeg>=0 && gxbeg+(ng-1)*jgx<nx1 && gzbeg+(ng-1)*jgz<nz1))	
    { printf("geophones exceeds the computing zone!"); exit(1);}
    if (csdgather)	
    {
        distx = sxbeg - gxbeg;
        distz = szbeg - gzbeg;
        if (!( sxbeg+(ns-1)*jsx+(ng-1)*jgx-distx <nx1  && szbeg+(ns-1)*jsz+(ng-1)*jgz-distz <nz1))	
        { printf("geophones exceeds the computing zone!"); exit(1);}
    }
    cuda_set_sg<<<(ng+255)/256, 256>>>(d_Gxz, gxbeg, gzbeg, jgx, jgz, ng, npml, nnz);
    
    cudaMemcpy(d_d1z, d1z, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d2x, d2x, nnx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vp,  vp,  M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs,  vs,  M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, rho, M, cudaMemcpyHostToDevice);
    
    for (is = 0; is < ns; is++)
    {
        cudaEventRecord(start);
        cudaMemset(d_vx,  0, M);   cudaMemset(d_vxx,  0, M);   cudaMemset(d_vxz,  0, M);
        cudaMemset(d_vz,  0, M);   cudaMemset(d_vzx,  0, M);   cudaMemset(d_vzz,  0, M);
        cudaMemset(d_txx, 0, M);   cudaMemset(d_txxx, 0, M);   cudaMemset(d_txxz, 0, M);
        cudaMemset(d_tzz, 0, M);   cudaMemset(d_tzzx, 0, M);   cudaMemset(d_tzzz, 0, M);
        cudaMemset(d_txz, 0, M);   cudaMemset(d_txzx, 0, M);   cudaMemset(d_txzz, 0, M);
        cudaMemset(d_vxp, 0, M);   cudaMemset(d_vxs,  0, M);   cudaMemset(d_vzp,  0, M);   cudaMemset(d_vzs,  0, M);
        for (it = 0; it < nt; it ++)
        {
            if (it < 200)
            {
                cuda_add_source<<<1, 1>>>(d_vz, &d_wlt[it], &d_Sxz[is], ns, true);
                // cuda_add_source<<<1, 1>>>(d_vx, &d_wlt[it], &d_Sxz[is], ns, true);
            }
            cuda_forward_vel<<<(N + 511) / 512, 512>>>(d_vx, d_vxx, d_vxz, d_vz, d_vzx, d_vzz, d_vxp, d_vzp, d_vxs, d_vzs, d_txx, d_txz, d_tzz, 
                                                        d_d2x, d_d1z, d_rho, d_vp, d_vs, nnz, nnx, dt, dx, dz, npml);
            cuda_forward_stress<<<(N + 511) / 512, 512>>>(d_txx, d_txxx, d_txxz, d_tzz, d_tzzx, d_tzzz, d_txz, d_txzx, d_txzz, d_vx, d_vz, 
                                                        d_d2x, d_d1z, d_rho, d_vp, d_vs, nnz, nnx, dt, dx, dz, npml);

            // cuda_forward_vel<<<(N + 511) / 512, 512>>>(d_vx, d_vxx, d_vxz, d_vz, d_vzx, d_vzz, d_vxp, d_vzp, d_vxs, d_vzs, d_txx, d_txz, d_tzz, 
            //                                             d_d2x, d_d1z, d_rho, d_vp, d_vs, nnz, nnx, dt, dx, dz);
            // cuda_forward_stress<<<(N + 511) / 512, 512>>>(d_txx, d_txxx, d_txxz, d_tzz, d_tzzx, d_tzzz, d_txz, d_txzx, d_txzz, d_vx, d_vz, 
            //                                             d_d2x, d_d1z, d_rho, d_vp, d_vs, nnz, nnx, dt, dx, dz);

            cuda_record<<<(ng + 255) / 256, 256>>>(&d_obsvx[it * ng], d_vx, d_Gxz, ng); 
            cuda_record<<<(ng + 255) / 256, 256>>>(&d_obsvz[it * ng], d_vz, d_Gxz, ng); 
                 
            if ((it == 500) && (is == 0) && it != 0)
            {
                printf("it = %d \n", it);
                cudaMemcpy(h_snopshot, d_vx, M, cudaMemcpyDeviceToHost);
                window(v0, h_snopshot, npml, nnz, nnx, nz1, nx1);
                fwrite(v0, 4L, nz1*nx1, fpsnopvx);
                
                cudaMemcpy(h_snopshot, d_vz, M, cudaMemcpyDeviceToHost);
                window(v0, h_snopshot, npml, nnz, nnx, nz1, nx1);
                fwrite(v0, 4L, nz1*nx1, fpsnopvz);

                cudaMemcpy(h_snopshot, d_vxp, M, cudaMemcpyDeviceToHost);
                window(v0, h_snopshot, npml, nnz, nnx, nz1, nx1);
                fwrite(v0, 4L, nz1*nx1, fpsnopvxp);

                cudaMemcpy(h_snopshot, d_vxs, M, cudaMemcpyDeviceToHost);
                window(v0, h_snopshot, npml, nnz, nnx, nz1, nx1);
                fwrite(v0, 4L, nz1*nx1, fpsnopvxs);
            }
            cudaError_t cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) 
            {
                printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }
        }
        
        cudaMemcpy(seis, d_obsvx, ng * nt * sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(seis, 4L, ng * nt, fpshotvx);
        cudaMemcpy(seis, d_obsvz, ng * nt * sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(seis, 4L, ng * nt, fpshotvz);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("%d shot finished: %g (s). \n",is+1, elapsed_time*1.e-3);

    }//is loop end

    /* <close FILE> */
    fclose(fpsnopvx); 
    fclose(fpsnopvz); 
    fclose(fpsnopvxp);
    fclose(fpsnopvxs);
    fclose(fpshotvx);
    fclose(fpshotvz);

    printf("end\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /* host memory free */
    free(seis);
    free(v0);
    free(vp);  free(vs);  free(rho);
    free(d1z); free(d2x);
    /* device memory free */
    cudaFree(d_Sxz); cudaFree(d_Gxz);
    cudaFree(d_d1z); cudaFree(d_d2x);
    cudaFree(d_wlt); cudaFree(d_obsvx);cudaFree(d_obsvz);  cudaFree(d_boundary_vx); cudaFree(d_boundary_vz);
    cudaFree(d_vp);  cudaFree(d_vs);   cudaFree(d_rho);
    cudaFree(d_vx);  cudaFree(d_vxx);  cudaFree(d_vxz);
    cudaFree(d_vz);  cudaFree(d_vzx);  cudaFree(d_vzz);
    cudaFree(d_txx); cudaFree(d_txxx); cudaFree(d_txxz);
    cudaFree(d_tzz); cudaFree(d_tzzx); cudaFree(d_tzzz);
    cudaFree(d_txz); cudaFree(d_txzx); cudaFree(d_txzz);
    cudaFree(d_vxp); cudaFree(d_vxs);  cudaFree(d_vzp);   cudaFree(d_vzs);
    return 0;
}

