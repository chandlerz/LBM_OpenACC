#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sys/time.h>

#define Q   15
#define D   3
#define Nx  64
#define Ny  32
#define Nz  32
#define IND1D(x, y, z) ((z*Ny*Nx) + (y*Nx) + x)

#define _OpenACC_
//#define _CUDA_

using namespace std;

void init();
void propagate();
void collision();
void save_flow_field(int t);
void compute_x_slide(int xid, double *prhoAve, double *uAve, double *vAve, double *wAve);

#ifdef _CUDA_
__global__ void LBM3D_propagate(double *f00, double *f01, double *f02, double *f03, double *f04,
                                double *f05, double *f06, double *f07, double *f08, double *f09,
                                double *f10, double *f11, double *f12, double *f13, double *f14,
                                int *flag,
                                double *f00temp, double *f01temp, double *f02temp, double *f03temp, double *f04temp,
                                double *f05temp, double *f06temp, double *f07temp, double *f08temp, double *f09temp,
                                double *f10temp, double *f11temp, double *f12temp, double *f13temp, double *f14temp);

__global__ void LBM3D_collision(double *f00, double *f01, double *f02, double *f03, double *f04,
                                double *f05, double *f06, double *f07, double *f08, double *f09,
                                double *f10, double *f11, double *f12, double *f13, double *f14,
                                int *flag,
                                double tau, double gx, double gy, double gz);
#define checkCudaAPIErrors(F) if ((F) != cudaSuccess) \
{ printf("Error at line %d in file %s: %s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError())); exit(-1); }

#endif //_CUDA_

const int t_max=5000;
const double rho0=1.0;

// Gravity driven
const double Gx=3.0e-4;
const double Gy=0.0;
const double Gz=0.0;

enum{FLUID, STATICWALL, MOVINGWALL, PERIODIC};

int e[Q][D]={{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},
			{0,0,1},{0,0,-1},{1,1,1},{-1,-1,-1},{1,1,-1},
			{-1,-1,1},{1,-1,1},{-1,1,-1},{1,-1,-1},{-1,1,1}};

int flag[Nz][Ny][Nx];
double tau, mu;
double w[Q]={2.0/9,
			1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,
			1.0/72,1.0/72,1.0/72,1.0/72,1.0/72,1.0/72,1.0/72,1.0/72};
double f0[Nz][Ny][Nx],f1[Nz][Ny][Nx],f2[Nz][Ny][Nx],f3[Nz][Ny][Nx],f4[Nz][Ny][Nx],
	   f5[Nz][Ny][Nx],f6[Nz][Ny][Nx],f7[Nz][Ny][Nx],f8[Nz][Ny][Nx],f9[Nz][Ny][Nx],
	   f10[Nz][Ny][Nx],f11[Nz][Ny][Nx],f12[Nz][Ny][Nx],f13[Nz][Ny][Nx],f14[Nz][Ny][Nx];

double f0temp[Nz][Ny][Nx],f1temp[Nz][Ny][Nx],f2temp[Nz][Ny][Nx],f3temp[Nz][Ny][Nx],f4temp[Nz][Ny][Nx],
	   f5temp[Nz][Ny][Nx],f6temp[Nz][Ny][Nx],f7temp[Nz][Ny][Nx],f8temp[Nz][Ny][Nx],f9temp[Nz][Ny][Nx],
	   f10temp[Nz][Ny][Nx],f11temp[Nz][Ny][Nx],f12temp[Nz][Ny][Nx],f13temp[Nz][Ny][Nx],f14temp[Nz][Ny][Nx];
char flowPath[30]="Flow/",FlowPath[60];
ofstream outdata;

int main()
{
	int hop=500;
    double rhoAve_x, uAve, vAve, wAve;
    struct timeval start;
    struct timeval end;
    double elapsedTime;

	tau=1.0;
	mu=(2*tau-1.0)/6.0;

	/*************Initializing with rho=rho0, u=v=0******************/
	cout<<"Initializing..."<<endl;

	init();
	cout<<"Complete initialization"<<endl;

	/********************Computing***********************************/
	cout<<"Computing..."<<endl;

    // timer begin
    gettimeofday(&start, NULL);

#ifdef _OpenACC_
#pragma acc data copyin(flag) \
                 copy(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14)    \
                 create(f0temp, f1temp, f2temp,  f3temp,  f4temp,  f5temp,  f6temp, f7temp, \
                        f8temp, f9temp, f10temp, f11temp, f12temp, f13temp, f14temp) 
    {
        for (int t=1;t<=t_max;t++)
        {
            propagate();
            collision();
            if (t%hop==0)
            {
                cout<<"timestep = "<<t<<endl;
            }
        }
    }
#endif //_OpenACC_

#ifdef _CUDA_
    dim3 block(32, 16, 1);
    dim3 grid(Nx/block.x, Ny/block.y, Nz/block.z);

    double *d_f00, *d_f01, *d_f02, *d_f03, *d_f04;
    double *d_f05, *d_f06, *d_f07, *d_f08, *d_f09;
    double *d_f10, *d_f11, *d_f12, *d_f13, *d_f14;

    int *d_flag;
    double *d_f00temp, *d_f01temp, *d_f02temp, *d_f03temp, *d_f04temp;
    double *d_f05temp, *d_f06temp, *d_f07temp, *d_f08temp, *d_f09temp;
    double *d_f10temp, *d_f11temp, *d_f12temp, *d_f13temp, *d_f14temp;

    int totalNodes = Nx*Ny*Nz;
    int totalBytes = totalNodes * sizeof(double);

    checkCudaAPIErrors(cudaMalloc((void **)&d_flag, totalNodes*sizeof(int)));

    checkCudaAPIErrors(cudaMalloc((void **)&d_f00, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f01, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f02, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f03, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f04, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f05, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f06, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f07, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f08, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f09, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f10, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f11, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f12, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f13, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f14, totalBytes));

    checkCudaAPIErrors(cudaMalloc((void **)&d_f00temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f01temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f02temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f03temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f04temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f05temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f06temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f07temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f08temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f09temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f10temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f11temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f12temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f13temp, totalBytes));
    checkCudaAPIErrors(cudaMalloc((void **)&d_f14temp, totalBytes));

    checkCudaAPIErrors(cudaMemcpy(d_flag,&flag[0][0][0], totalNodes*sizeof(int), cudaMemcpyHostToDevice));

    checkCudaAPIErrors(cudaMemcpy(d_f00, &f0[0][0][0],  totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f01, &f1[0][0][0],  totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f02, &f2[0][0][0],  totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f03, &f3[0][0][0],  totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f04, &f4[0][0][0],  totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f05, &f5[0][0][0],  totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f06, &f6[0][0][0],  totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f07, &f7[0][0][0],  totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f08, &f8[0][0][0],  totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f09, &f9[0][0][0],  totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f10, &f10[0][0][0], totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f11, &f11[0][0][0], totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f12, &f12[0][0][0], totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f13, &f13[0][0][0], totalBytes, cudaMemcpyHostToDevice));
    checkCudaAPIErrors(cudaMemcpy(d_f14, &f14[0][0][0], totalBytes, cudaMemcpyHostToDevice));

    for (int t=1;t<=t_max/2;t++)
    {

        LBM3D_propagate<<<grid, block>>>(d_f00, d_f01, d_f02, d_f03, d_f04,
                                         d_f05, d_f06, d_f07, d_f08, d_f09,
                                         d_f10, d_f11, d_f12, d_f13, d_f14,
                                         d_flag,
                                         d_f00temp, d_f01temp, d_f02temp, d_f03temp, d_f04temp,
                                         d_f05temp, d_f06temp, d_f07temp, d_f08temp, d_f09temp,
                                         d_f10temp, d_f11temp, d_f12temp, d_f13temp, d_f14temp);
        LBM3D_collision<<<grid, block>>>(d_f00temp, d_f01temp, d_f02temp, d_f03temp, d_f04temp,
                                         d_f05temp, d_f06temp, d_f07temp, d_f08temp, d_f09temp,
                                         d_f10temp, d_f11temp, d_f12temp, d_f13temp, d_f14temp,
                                         d_flag,
                                         tau, Gx, Gy, Gz);
        checkCudaAPIErrors(cudaDeviceSynchronize());

        LBM3D_propagate<<<grid, block>>>(d_f00temp, d_f01temp, d_f02temp, d_f03temp, d_f04temp,
                                         d_f05temp, d_f06temp, d_f07temp, d_f08temp, d_f09temp,
                                         d_f10temp, d_f11temp, d_f12temp, d_f13temp, d_f14temp,
                                         d_flag,
                                         d_f00, d_f01, d_f02, d_f03, d_f04,
                                         d_f05, d_f06, d_f07, d_f08, d_f09,
                                         d_f10, d_f11, d_f12, d_f13, d_f14);
        LBM3D_collision<<<grid, block>>>(d_f00, d_f01, d_f02, d_f03, d_f04,
                                         d_f05, d_f06, d_f07, d_f08, d_f09,
                                         d_f10, d_f11, d_f12, d_f13, d_f14,
                                         d_flag,
                                         tau, Gx, Gy, Gz);
        checkCudaAPIErrors(cudaDeviceSynchronize());

        if ((2*t)%hop==0)
        {
            cout<<"timestep = "<<2*t<<endl;
        }
    }

    checkCudaAPIErrors(cudaMemcpy(&f0[0][0][0],  d_f00, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f1[0][0][0],  d_f01, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f2[0][0][0],  d_f02, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f3[0][0][0],  d_f03, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f4[0][0][0],  d_f04, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f5[0][0][0],  d_f05, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f6[0][0][0],  d_f06, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f7[0][0][0],  d_f07, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f8[0][0][0],  d_f08, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f9[0][0][0],  d_f09, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f10[0][0][0], d_f10, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f11[0][0][0], d_f11, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f12[0][0][0], d_f12, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f13[0][0][0], d_f13, totalBytes, cudaMemcpyDeviceToHost));
    checkCudaAPIErrors(cudaMemcpy(&f14[0][0][0], d_f14, totalBytes, cudaMemcpyDeviceToHost));

    checkCudaAPIErrors(cudaFree(d_flag));

    checkCudaAPIErrors(cudaFree(d_f00));
    checkCudaAPIErrors(cudaFree(d_f01));
    checkCudaAPIErrors(cudaFree(d_f02));
    checkCudaAPIErrors(cudaFree(d_f03));
    checkCudaAPIErrors(cudaFree(d_f04));
    checkCudaAPIErrors(cudaFree(d_f05));
    checkCudaAPIErrors(cudaFree(d_f06));
    checkCudaAPIErrors(cudaFree(d_f07));
    checkCudaAPIErrors(cudaFree(d_f08));
    checkCudaAPIErrors(cudaFree(d_f09));
    checkCudaAPIErrors(cudaFree(d_f10));
    checkCudaAPIErrors(cudaFree(d_f11));
    checkCudaAPIErrors(cudaFree(d_f12));
    checkCudaAPIErrors(cudaFree(d_f13));
    checkCudaAPIErrors(cudaFree(d_f14));

    checkCudaAPIErrors(cudaFree(d_f00temp));
    checkCudaAPIErrors(cudaFree(d_f01temp));
    checkCudaAPIErrors(cudaFree(d_f02temp));
    checkCudaAPIErrors(cudaFree(d_f03temp));
    checkCudaAPIErrors(cudaFree(d_f04temp));
    checkCudaAPIErrors(cudaFree(d_f05temp));
    checkCudaAPIErrors(cudaFree(d_f06temp));
    checkCudaAPIErrors(cudaFree(d_f07temp));
    checkCudaAPIErrors(cudaFree(d_f08temp));
    checkCudaAPIErrors(cudaFree(d_f09temp));
    checkCudaAPIErrors(cudaFree(d_f10temp));
    checkCudaAPIErrors(cudaFree(d_f11temp));
    checkCudaAPIErrors(cudaFree(d_f12temp));
    checkCudaAPIErrors(cudaFree(d_f13temp));
    checkCudaAPIErrors(cudaFree(d_f14temp));

#endif //_CUDA_

    gettimeofday(&end, NULL);
    elapsedTime  = (end.tv_sec - start.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;  // us  to ms

    compute_x_slide(Nx-1, &rhoAve_x, &uAve, &vAve, &wAve);
    cout<<"The average rho at outlet is "<< rhoAve_x <<endl;
    cout<<"The average u at outlet is "<< uAve <<endl;
    cout<<"The average v at outlet is "<< vAve <<endl;
    cout<<"The average w at outlet is "<< wAve <<endl;
    printf("the elapsedTime is %f ms\n", elapsedTime);

	cout<<"Complete computation"<<endl;

	return 0;
}

void init()
{
	int x, y, z;
	double rho,vx,vy,vz;
	double square, product;

	for (z=0;z<Nz;z++)
	{
		for (y=0;y<Ny;y++)
		{
			for (x=0;x<Nx;x++)
			{
				rho=rho0;
				vx=0.0;
				vy=0.0;
				vz=0.0;

				square=1.5*(vx*vx+vy*vy+vz*vz);
				
				f0[z][y][x]=w[0]*rho*(1.0-square);
				
				f1[z][y][x]=w[1]*rho*(1.0+3.0*vx+4.5*vx*vx-square);
				f2[z][y][x]=f1[z][y][x]-6.0*w[1]*rho*vx;
				
				f3[z][y][x]=w[3]*rho*(1.0+3.0*vy+4.5*vy*vy-square);
				f4[z][y][x]=f3[z][y][x]-6.0*w[3]*rho*vy;
				
				f5[z][y][x]=w[5]*rho*(1.0+3.0*vz+4.5*vz*vz-square);
				f6[z][y][x]=f5[z][y][x]-6.0*w[5]*rho*vz;
				
				product=vx+vy+vz;
				f7[z][y][x]=w[7]*rho*(1.0+3.0*product+4.5*product*product-square);
				f8[z][y][x]=f7[z][y][x]-6.0*w[7]*rho*product;
				
				product=vx+vy-vz;
				f9[z][y][x]=w[9]*rho*(1.0+3.0*product+4.5*product*product-square);
				f10[z][y][x]=f9[z][y][x]-6.0*w[9]*rho*product;
				
				product=vx-vy+vz;
				f11[z][y][x]=w[11]*rho*(1.0+3.0*product+4.5*product*product-square);
				f12[z][y][x]=f11[z][y][x]-6.0*w[11]*rho*product;
				
				product=vx-vy-vz;
				f13[z][y][x]=w[13]*rho*(1.0+3.0*product+4.5*product*product-square);
				f14[z][y][x]=f13[z][y][x]-6.0*w[13]*rho*product;

				if (y==0||y==Ny-1||z==0||z==Nz-1)
				{
					flag[z][y][x]=STATICWALL;
				}
				else if (((x==0||x==Nx-1)&&(y>0&&y<Ny-1)&&(z>0&&z<Nz-1)))
				{
					flag[z][y][x]=PERIODIC;
				}
				else
					flag[z][y][x]=FLUID;
			}
		}
	}
}

void propagate()
{
	int x, y, z;
	int xp, yp, zp;
	int k;

#pragma acc parallel present(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14)    \
                     present(f0temp, f1temp, f2temp,  f3temp,  f4temp,  f5temp,  f6temp, f7temp, \
                             f8temp, f9temp, f10temp, f11temp, f12temp, f13temp, f14temp)        \
                     device_type(nvidia) num_workers(8) vector_length(Nx)
                     //device_type(nvidia) num_gangs(128) num_workers(8) vector_length(Nx)
#pragma acc loop device_type(nvidia) gang
//#pragma acc loop collapse(3) 
	for (z=0;z<Nz;z++)
	{
#pragma acc loop device_type(nvidia) worker
		for (y=0;y<Ny;y++)
		{
#pragma acc loop device_type(nvidia) vector
			for (x=0;x<Nx;x++)
			{
				if (FLUID==flag[z][y][x])
				{
					k=0;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f0temp[z][y][x]=f0[zp][yp][xp];

					k=1;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f1temp[z][y][x]=f1[zp][yp][xp];

					k=2;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f2temp[z][y][x]=f2[zp][yp][xp];

					k=3;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f3temp[z][y][x]=f3[zp][yp][xp];

					k=4;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f4temp[z][y][x]=f4[zp][yp][xp];

					k=5;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f5temp[z][y][x]=f5[zp][yp][xp];

					k=6;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f6temp[z][y][x]=f6[zp][yp][xp];

					k=7;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f7temp[z][y][x]=f7[zp][yp][xp];

					k=8;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f8temp[z][y][x]=f8[zp][yp][xp];

					k=9;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f9temp[z][y][x]=f9[zp][yp][xp];

					k=10;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f10temp[z][y][x]=f10[zp][yp][xp];

					k=11;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f11temp[z][y][x]=f11[zp][yp][xp];

					k=12;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f12temp[z][y][x]=f12[zp][yp][xp];

					k=13;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f13temp[z][y][x]=f13[zp][yp][xp];

					k=14;
					xp=x-e[k][0];
					yp=y-e[k][1];
					zp=z-e[k][2];
					f14temp[z][y][x]=f14[zp][yp][xp];
				}
				else
				{
					k=0;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f0temp[z][y][x]=f0[zp][yp][xp];
					
					k=1;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f1temp[z][y][x]=f1[zp][yp][xp];
					
					k=2;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f2temp[z][y][x]=f2[zp][yp][xp];
					
					k=3;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f3temp[z][y][x]=f3[zp][yp][xp];
					
					k=4;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f4temp[z][y][x]=f4[zp][yp][xp];
					
					k=5;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f5temp[z][y][x]=f5[zp][yp][xp];
					
					k=6;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f6temp[z][y][x]=f6[zp][yp][xp];
					
					k=7;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f7temp[z][y][x]=f7[zp][yp][xp];
					
					k=8;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f8temp[z][y][x]=f8[zp][yp][xp];
					
					k=9;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f9temp[z][y][x]=f9[zp][yp][xp];
					
					k=10;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f10temp[z][y][x]=f10[zp][yp][xp];
					
					k=11;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f11temp[z][y][x]=f11[zp][yp][xp];
					
					k=12;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f12temp[z][y][x]=f12[zp][yp][xp];
					
					k=13;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f13temp[z][y][x]=f13[zp][yp][xp];
					
					k=14;
					xp=(x-e[k][0]+Nx)%Nx;
					yp=(y-e[k][1]+Ny)%Ny;
					zp=(z-e[k][2]+Nz)%Nz;
					f14temp[z][y][x]=f14[zp][yp][xp];
				}
			}
		}
	}
}

void collision()
{
	int x, y, z;
	double rho,vx,vy,vz;
	double f_eq0,f_eq1,f_eq2,f_eq3,f_eq4,f_eq5,f_eq6,f_eq7,f_eq8,f_eq9,f_eq10,f_eq11,f_eq12,f_eq13,f_eq14;
	double square, tau_inv, dummy, product;

	tau_inv=1.0/tau;

#pragma acc parallel present(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14)    \
                     present(f0temp, f1temp, f2temp,  f3temp,  f4temp,  f5temp,  f6temp, f7temp, \
                             f8temp, f9temp, f10temp, f11temp, f12temp, f13temp, f14temp)        \
                     device_type(nvidia)  num_workers(8) vector_length(Nx)
                     //device_type(nvidia) num_gangs(128) num_workers(8) vector_length(Nx)
//#pragma acc loop collapse(3) 
#pragma acc loop device_type(nvidia) gang
	for (z=0;z<Nz;z++)
	{
#pragma acc loop device_type(nvidia) worker
		for (y=0;y<Ny;y++)
		{
#pragma acc loop device_type(nvidia) vector
			for (x=0;x<Nx;x++)
			{
				f0[z][y][x]=f0temp[z][y][x];f1[z][y][x]=f1temp[z][y][x];f2[z][y][x]=f2temp[z][y][x];
				f3[z][y][x]=f3temp[z][y][x];f4[z][y][x]=f4temp[z][y][x];f5[z][y][x]=f5temp[z][y][x];
				f6[z][y][x]=f6temp[z][y][x];f7[z][y][x]=f7temp[z][y][x];f8[z][y][x]=f8temp[z][y][x];
				f9[z][y][x]=f9temp[z][y][x];f10[z][y][x]=f10temp[z][y][x];f11[z][y][x]=f11temp[z][y][x];
				f12[z][y][x]=f12temp[z][y][x];f13[z][y][x]=f13temp[z][y][x];f14[z][y][x]=f14temp[z][y][x];

				if (STATICWALL==flag[z][y][x])
				{
					dummy=f1[z][y][x];
					f1[z][y][x]=f2[z][y][x];
					f2[z][y][x]=dummy;

					dummy=f3[z][y][x];
					f3[z][y][x]=f4[z][y][x];
					f4[z][y][x]=dummy;

					dummy=f5[z][y][x];
					f5[z][y][x]=f6[z][y][x];
					f6[z][y][x]=dummy;

					dummy=f7[z][y][x];
					f7[z][y][x]=f8[z][y][x];
					f8[z][y][x]=dummy;

					dummy=f9[z][y][x];
					f9[z][y][x]=f10[z][y][x];
					f10[z][y][x]=dummy;

					dummy=f11[z][y][x];
					f11[z][y][x]=f12[z][y][x];
					f12[z][y][x]=dummy;

					dummy=f13[z][y][x];
					f13[z][y][x]=f14[z][y][x];
					f14[z][y][x]=dummy;
				}
				else
				{
					rho=f0[z][y][x]+f1[z][y][x]+f2[z][y][x]+f3[z][y][x]+f4[z][y][x]
						+f5[z][y][x]+f6[z][y][x]+f7[z][y][x]+f8[z][y][x]+f9[z][y][x]
						+f10[z][y][x]+f11[z][y][x]+f12[z][y][x]+f13[z][y][x]+f14[z][y][x];
					vx=(f1[z][y][x]-f2[z][y][x]+f7[z][y][x]-f8[z][y][x]+f9[z][y][x]
						-f10[z][y][x]+f11[z][y][x]-f12[z][y][x]+f13[z][y][x]-f14[z][y][x])/rho;
					vy=(f3[z][y][x]-f4[z][y][x]+f7[z][y][x]-f8[z][y][x]+f9[z][y][x]
						-f10[z][y][x]+f12[z][y][x]-f11[z][y][x]+f14[z][y][x]-f13[z][y][x])/rho;
					vz=(f5[z][y][x]-f6[z][y][x]+f7[z][y][x]-f8[z][y][x]+f10[z][y][x]
						-f9[z][y][x]+f11[z][y][x]-f12[z][y][x]+f14[z][y][x]-f13[z][y][x])/rho;

					vx=vx+tau*Gx;
					vy=vy+tau*Gy;
					vz=vz+tau*Gz;
					
					square=1.5*(vx*vx+vy*vy+vz*vz);
					
					f_eq0=w[0]*rho*(1.0-square);
					
					f_eq1=w[1]*rho*(1.0+3.0*vx+4.5*vx*vx-square);
					f_eq2=f_eq1-6.0*w[1]*rho*vx;
					
					f_eq3=w[3]*rho*(1.0+3.0*vy+4.5*vy*vy-square);
					f_eq4=f_eq3-6.0*w[3]*rho*vy;
					
					f_eq5=w[5]*rho*(1.0+3.0*vz+4.5*vz*vz-square);
					f_eq6=f_eq5-6.0*w[5]*rho*vz;
					
					product=vx+vy+vz;
					f_eq7=w[7]*rho*(1.0+3.0*product+4.5*product*product-square);
					f_eq8=f_eq7-6.0*w[7]*rho*product;
					
					product=vx+vy-vz;
					f_eq9=w[9]*rho*(1.0+3.0*product+4.5*product*product-square);
					f_eq10=f_eq9-6.0*w[9]*rho*product;
					
					product=vx-vy+vz;
					f_eq11=w[11]*rho*(1.0+3.0*product+4.5*product*product-square);
					f_eq12=f_eq11-6.0*w[11]*rho*product;
					
					product=vx-vy-vz;
					f_eq13=w[13]*rho*(1.0+3.0*product+4.5*product*product-square);
					f_eq14=f_eq13-6.0*w[13]*rho*product;
					
					f0[z][y][x]+=(f_eq0-f0[z][y][x])*tau_inv;
					f1[z][y][x]+=(f_eq1-f1[z][y][x])*tau_inv;
					f2[z][y][x]+=(f_eq2-f2[z][y][x])*tau_inv;
					f3[z][y][x]+=(f_eq3-f3[z][y][x])*tau_inv;
					f4[z][y][x]+=(f_eq4-f4[z][y][x])*tau_inv;
					f5[z][y][x]+=(f_eq5-f5[z][y][x])*tau_inv;
					f6[z][y][x]+=(f_eq6-f6[z][y][x])*tau_inv;
					f7[z][y][x]+=(f_eq7-f7[z][y][x])*tau_inv;
					f8[z][y][x]+=(f_eq8-f8[z][y][x])*tau_inv;
					f9[z][y][x]+=(f_eq9-f9[z][y][x])*tau_inv;
					f10[z][y][x]+=(f_eq10-f10[z][y][x])*tau_inv;
					f11[z][y][x]+=(f_eq11-f11[z][y][x])*tau_inv;
					f12[z][y][x]+=(f_eq12-f12[z][y][x])*tau_inv;
					f13[z][y][x]+=(f_eq13-f13[z][y][x])*tau_inv;
					f14[z][y][x]+=(f_eq14-f14[z][y][x])*tau_inv;
				}
			}
		}
	}
}

void save_flow_field(int t)
{
	int x, y, z;
	double rho;
	double vx, vy, vz, v_mod;
	char tname1[30],tname2[30];
	stringstream tstring;
	tstring<<t;
	tstring>>tname2;
	strcpy(tname1,tname2);
   	strcpy(FlowPath,flowPath);
   	strcat(strcat(FlowPath,tname1),".txt");
	
	outdata.open(FlowPath,ios::out);
	if (!outdata)
	{
		cout<<"Can not open file!"<<endl;
	}
	outdata<< "Title= \"3D Duct Flow using LBM\""<<endl;
	outdata<< "VARIABLES=\"X\",\"Y\",\"Z\",\"Vx\",\"Vy\",\"Vz\",\"V\",\"RHO\""<<endl;
	outdata<< "ZONE T= \"BOX\",I="<< Nx<<",J="<< Ny<<",K="<< Nz<<",F=POINT"<<endl;
	
	for (z=0;z<Nz;z++)
	{
		for(y=0;y<Ny;y++)
		{
			for(x=0;x<Nx;x++)
			{
				rho=f0[z][y][x]+f1[z][y][x]+f2[z][y][x]+f3[z][y][x]+f4[z][y][x]
				   +f5[z][y][x]+f6[z][y][x]+f7[z][y][x]+f8[z][y][x]+f9[z][y][x]
				   +f10[z][y][x]+f11[z][y][x]+f12[z][y][x]+f13[z][y][x]+f14[z][y][x];
				vx=(f1[z][y][x]-f2[z][y][x]+f7[z][y][x]-f8[z][y][x]+f9[z][y][x]
				   -f10[z][y][x]+f11[z][y][x]-f12[z][y][x]+f13[z][y][x]-f14[z][y][x])/rho;
				vy=(f3[z][y][x]-f4[z][y][x]+f7[z][y][x]-f8[z][y][x]+f9[z][y][x]
				   -f10[z][y][x]+f12[z][y][x]-f11[z][y][x]+f14[z][y][x]-f13[z][y][x])/rho;
				vz=(f5[z][y][x]-f6[z][y][x]+f7[z][y][x]-f8[z][y][x]+f10[z][y][x]
				   -f9[z][y][x]+f11[z][y][x]-f12[z][y][x]+f14[z][y][x]-f13[z][y][x])/rho;
				v_mod=sqrt(vx*vx+vy*vy+vz*vz);
				outdata<<double(x)<<"	"<<double(y)<<"	"<<double(z)<<"	"
					   <<vx<<"	"<<vy<<"	"<<vz<<"	"<<v_mod<<"	"<<rho<<endl;
			}
		}
	}	
	outdata.close();
}

void compute_x_slide(int xid, double *prhoAve, double *uAve, double *vAve, double *wAve)
{
    int x, y, z;
    double rho, u, v, w;
    double rhoSum = 0.0, uSum = 0.0;
    double vSum = 0.0, wSum = 0.0;

    x = xid;

    for (z=0;z<Nz;z++)
    {
        for(y=0;y<Ny;y++)
        {
            rho=f0[z][y][x]+f1[z][y][x]+f2[z][y][x]+f3[z][y][x]+f4[z][y][x]
               +f5[z][y][x]+f6[z][y][x]+f7[z][y][x]+f8[z][y][x]+f9[z][y][x]
               +f10[z][y][x]+f11[z][y][x]+f12[z][y][x]+f13[z][y][x]+f14[z][y][x];
		    u =(f1[z][y][x]-f2[z][y][x]+f7[z][y][x]-f8[z][y][x]+f9[z][y][x]
		       -f10[z][y][x]+f11[z][y][x]-f12[z][y][x]+f13[z][y][x]-f14[z][y][x])/rho;
		    v =(f3[z][y][x]-f4[z][y][x]+f7[z][y][x]-f8[z][y][x]+f9[z][y][x]
		       -f10[z][y][x]+f12[z][y][x]-f11[z][y][x]+f14[z][y][x]-f13[z][y][x])/rho;
		    w =(f5[z][y][x]-f6[z][y][x]+f7[z][y][x]-f8[z][y][x]+f10[z][y][x]
		       -f9[z][y][x]+f11[z][y][x]-f12[z][y][x]+f14[z][y][x]-f13[z][y][x])/rho;

            rhoSum += rho;
            uSum   += u;
            vSum   += v;
            wSum   += w;
        }
    }

    *prhoAve = (rhoSum/(Ny*Nz));
    *uAve    = (uSum/(Ny*Nz));
    *vAve    = (vSum/(Ny*Nz));
    *wAve    = (wSum/(Ny*Nz));
}


#ifdef _CUDA_
__global__ void LBM3D_propagate(double *f00, double *f01, double *f02, double *f03, double *f04,
                                double *f05, double *f06, double *f07, double *f08, double *f09,
                                double *f10, double *f11, double *f12, double *f13, double *f14,
                                int *flag,
                                double *f00temp, double *f01temp, double *f02temp, double *f03temp, double *f04temp,
                                double *f05temp, double *f06temp, double *f07temp, double *f08temp, double *f09temp,
                                double *f10temp, double *f11temp, double *f12temp, double *f13temp, double *f14temp)
{
    int tag;
    int xS, yS, zS; //S: source
    int xD, yD, zD; //S: destination
    int destinationID;

    xD = threadIdx.x + blockDim.x * blockIdx.x;
    yD = threadIdx.y + blockDim.y * blockIdx.y;
    zD = threadIdx.z + blockDim.z * blockIdx.z;

    destinationID = IND1D(xD, yD, zD);

    tag = flag[destinationID];

    if (tag == FLUID)
    {
        //e(0, 0, 0)
        xS = xD - 0;
        yS = yD - 0;
        zS = zD - 0;
        f00temp[destinationID] = f00[IND1D(xS, yS, zS)]; 

        //e(1, 0, 0)
        xS = xD - 1;
        yS = yD - 0;
        zS = zD - 0;
        f01temp[destinationID] = f01[IND1D(xS, yS, zS)]; 

        //e(-1, 0, 0)
        xS = xD + 1;
        yS = yD - 0;
        zS = zD - 0;
        f02temp[destinationID] = f02[IND1D(xS, yS, zS)]; 

        //e(0, 1, 0)
        xS = xD - 0;
        yS = yD - 1;
        zS = zD - 0;
        f03temp[destinationID] = f03[IND1D(xS, yS, zS)]; 

        //e(0, -1, 0)
        xS = xD - 0;
        yS = yD + 1;
        zS = zD - 0;
        f04temp[destinationID] = f04[IND1D(xS, yS, zS)]; 

        //e(0, 0, 1)
        xS = xD - 0;
        yS = yD - 0;
        zS = zD - 1;
        f05temp[destinationID] = f05[IND1D(xS, yS, zS)]; 

        //e(0, 0, -1)
        xS = xD - 0;
        yS = yD - 0;
        zS = zD + 1;
        f06temp[destinationID] = f06[IND1D(xS, yS, zS)]; 

        //e(1, 1, 1)
        xS = xD - 1;
        yS = yD - 1;
        zS = zD - 1;
        f07temp[destinationID] = f07[IND1D(xS, yS, zS)]; 

        //e(-1, -1, -1)
        xS = xD + 1;
        yS = yD + 1;
        zS = zD + 1;
        f08temp[destinationID] = f08[IND1D(xS, yS, zS)]; 

        //e(1, 1, -1)
        xS = xD - 1;
        yS = yD - 1;
        zS = zD + 1;
        f09temp[destinationID] = f09[IND1D(xS, yS, zS)]; 

        //e(-1, -1, 1)
        xS = xD + 1;
        yS = yD + 1;
        zS = zD - 1;
        f10temp[destinationID] = f10[IND1D(xS, yS, zS)]; 

        //e(1, -1, 1)
        xS = xD - 1;
        yS = yD + 1;
        zS = zD - 1;
        f11temp[destinationID] = f11[IND1D(xS, yS, zS)]; 

        //e(-1, 1, -1)
        xS = xD + 1;
        yS = yD - 1;
        zS = zD + 1;
        f12temp[destinationID] = f12[IND1D(xS, yS, zS)]; 

        //e(1, -1, -1)
        xS = xD - 1;
        yS = yD + 1;
        zS = zD + 1;
        f13temp[destinationID] = f13[IND1D(xS, yS, zS)]; 

        //e(-1, 1, 1)
        xS = xD + 1;
        yS = yD - 1;
        zS = zD - 1;
        f14temp[destinationID] = f14[IND1D(xS, yS, zS)]; 
    }
    else
    {
        //e(0, 0, 0)
        xS = (xD - 0 + Nx) % Nx;
        yS = (yD - 0 + Ny) % Ny;
        zS = (zD - 0 + Nz) % Nz;
        f00temp[destinationID] = f00[IND1D(xS, yS, zS)]; 

        //e(1, 0, 0)
        xS = (xD - 1 + Nx) % Nx;
        yS = (yD - 0 + Ny) % Ny;
        zS = (zD - 0 + Nz) % Nz;
        f01temp[destinationID] = f01[IND1D(xS, yS, zS)]; 

        //e(-1, 0, 0)
        xS = (xD + 1 + Nx) % Nx;
        yS = (yD - 0 + Ny) % Ny;
        zS = (zD - 0 + Nz) % Nz;
        f02temp[destinationID] = f02[IND1D(xS, yS, zS)]; 

        //e(0, 1, 0)
        xS = (xD - 0 + Nx) % Nx;
        yS = (yD - 1 + Ny) % Ny;
        zS = (zD - 0 + Nz) % Nz;
        f03temp[destinationID] = f03[IND1D(xS, yS, zS)]; 

        //e(0, -1, 0)
        xS = (xD - 0 + Nx) % Nx;
        yS = (yD + 1 + Ny) % Ny;
        zS = (zD - 0 + Nz) % Nz;
        f04temp[destinationID] = f04[IND1D(xS, yS, zS)]; 

        //e(0, 0, 1)
        xS = (xD - 0 + Nx) % Nx;
        yS = (yD - 0 + Ny) % Ny;
        zS = (zD - 1 + Nz) % Nz;
        f05temp[destinationID] = f05[IND1D(xS, yS, zS)]; 

        //e(0, 0, -1)
        xS = (xD - 0 + Nx) % Nx;
        yS = (yD - 0 + Ny) % Ny;
        zS = (zD + 1 + Nz) % Nz;
        f06temp[destinationID] = f06[IND1D(xS, yS, zS)]; 

        //e(1, 1, 1)
        xS = (xD - 1 + Nx) % Nx;
        yS = (yD - 1 + Ny) % Ny;
        zS = (zD - 1 + Nz) % Nz;
        f07temp[destinationID] = f07[IND1D(xS, yS, zS)]; 

        //e(-1, -1, -1)
        xS = (xD + 1 + Nx) % Nx;
        yS = (yD + 1 + Ny) % Ny;
        zS = (zD + 1 + Nz) % Nz;
        f08temp[destinationID] = f08[IND1D(xS, yS, zS)]; 

        //e(1, 1, -1)
        xS = (xD - 1 + Nx) % Nx;
        yS = (yD - 1 + Ny) % Ny;
        zS = (zD + 1 + Nz) % Nz;
        f09temp[destinationID] = f09[IND1D(xS, yS, zS)]; 

        //e(-1, -1, 1)
        xS = (xD + 1 + Nx) % Nx;
        yS = (yD + 1 + Ny) % Ny;
        zS = (zD - 1 + Nz) % Nz;
        f10temp[destinationID] = f10[IND1D(xS, yS, zS)]; 

        //e(1, -1, 1)
        xS = (xD - 1 + Nx) % Nx;
        yS = (yD + 1 + Ny) % Ny;
        zS = (zD - 1 + Nz) % Nz;
        f11temp[destinationID] = f11[IND1D(xS, yS, zS)]; 

        //e(-1, 1, -1)
        xS = (xD + 1 + Nx) % Nx;
        yS = (yD - 1 + Ny) % Ny;
        zS = (zD + 1 + Nz) % Nz;
        f12temp[destinationID] = f12[IND1D(xS, yS, zS)]; 

        //e(1, -1, -1)
        xS = (xD - 1 + Nx) % Nx;
        yS = (yD + 1 + Ny) % Ny;
        zS = (zD + 1 + Nz) % Nz;
        f13temp[destinationID] = f13[IND1D(xS, yS, zS)]; 

        //e(-1, 1, 1)
        xS = (xD + 1 + Nx) % Nx;
        yS = (yD - 1 + Ny) % Ny;
        zS = (zD - 1 + Nz) % Nz;
        f14temp[destinationID] = f14[IND1D(xS, yS, zS)]; 
    }
}

__global__ void LBM3D_collision(double *f00, double *f01, double *f02, double *f03, double *f04,
                                double *f05, double *f06, double *f07, double *f08, double *f09,
                                double *f10, double *f11, double *f12, double *f13, double *f14,
                                int *flag,
                                double tau, double gx, double gy, double gz)
{
    int tag;
    int xD, yD, zD; //S: destination
    int destinationID;
    double f_eq00, f_eq01, f_eq02, f_eq03, f_eq04;
    double f_eq05, f_eq06, f_eq07, f_eq08, f_eq09;
    double f_eq10, f_eq11, f_eq12, f_eq13, f_eq14;

    double ftemp_00, ftemp_01, ftemp_02, ftemp_03, ftemp_04;
    double ftemp_05, ftemp_06, ftemp_07, ftemp_08, ftemp_09;
    double ftemp_10, ftemp_11, ftemp_12, ftemp_13, ftemp_14;

	double rho, vx, vy, vz, tau_inv;
	double square, dummy, product;

    xD = threadIdx.x + blockDim.x * blockIdx.x;
    yD = threadIdx.y + blockDim.y * blockIdx.y;
    zD = threadIdx.z + blockDim.z * blockIdx.z;

    destinationID = IND1D(xD, yD, zD);

    tau_inv = 1.0/tau;
    tag = flag[destinationID];

    if (tag == STATICWALL)
    {
        dummy = f01[destinationID]; 
        f01[destinationID] = f02[destinationID];
        f02[destinationID] = dummy;

        dummy = f03[destinationID]; 
        f03[destinationID] = f04[destinationID];
        f04[destinationID] = dummy;

        dummy = f05[destinationID]; 
        f05[destinationID] = f06[destinationID];
        f06[destinationID] = dummy;

        dummy = f07[destinationID]; 
        f07[destinationID] = f08[destinationID];
        f08[destinationID] = dummy;

        dummy = f09[destinationID]; 
        f09[destinationID] = f10[destinationID];
        f10[destinationID] = dummy;

        dummy = f11[destinationID]; 
        f11[destinationID] = f12[destinationID];
        f12[destinationID] = dummy;

        dummy = f13[destinationID]; 
        f13[destinationID] = f14[destinationID];
        f14[destinationID] = dummy;
    }
    else
    {
        ftemp_00 = f00[destinationID];
        ftemp_01 = f01[destinationID];
        ftemp_02 = f02[destinationID];
        ftemp_03 = f03[destinationID];
        ftemp_04 = f04[destinationID];
        ftemp_05 = f05[destinationID];
        ftemp_06 = f06[destinationID];
        ftemp_07 = f07[destinationID];
        ftemp_08 = f08[destinationID];
        ftemp_09 = f09[destinationID];
        ftemp_10 = f10[destinationID];
        ftemp_11 = f11[destinationID];
        ftemp_12 = f12[destinationID];
        ftemp_13 = f13[destinationID];
        ftemp_14 = f14[destinationID];

        rho = ftemp_00 + ftemp_01 + ftemp_02 + ftemp_03 + ftemp_04
            + ftemp_05 + ftemp_06 + ftemp_07 + ftemp_08 + ftemp_09
            + ftemp_10 + ftemp_11 + ftemp_12 + ftemp_13 + ftemp_14;

        vx = (ftemp_01-ftemp_02) + (ftemp_07-ftemp_08) +
             (ftemp_09-ftemp_10) + (ftemp_11-ftemp_12) +
             (ftemp_13-ftemp_14);
        vx /= rho;

        vy = (ftemp_03-ftemp_04) + (ftemp_07-ftemp_08) +
             (ftemp_09-ftemp_10) + (ftemp_12-ftemp_11) +
             (ftemp_14-ftemp_13);
        vy /= rho;
    
        vz = (ftemp_05-ftemp_06) + (ftemp_07-ftemp_08) +
             (ftemp_10-ftemp_09) + (ftemp_11-ftemp_12) +
             (ftemp_14-ftemp_13);
        vz /= rho;
            
        vx = vx + tau * gx;
        vy = vy + tau * gy;
        vz = vz + tau * gz;

        square=1.5*(vx*vx+vy*vy+vz*vz);

        f_eq00=(2.0/9)*rho*(1.0-square);

        f_eq01=(1.0/9)*rho*(1.0+3.0*vx+4.5*vx*vx-square);
        f_eq02=f_eq01-6.0*(1.0/9)*rho*vx;

        f_eq03=(1.0/9)*rho*(1.0+3.0*vy+4.5*vy*vy-square);
        f_eq04=f_eq03-6.0*(1.0/9)*rho*vy;

        f_eq05=(1.0/9)*rho*(1.0+3.0*vz+4.5*vz*vz-square);
        f_eq06=f_eq05-6.0*(1.0/9)*rho*vz;

        product=vx+vy+vz;
        f_eq07=(1.0/72)*rho*(1.0+3.0*product+4.5*product*product-square);
        f_eq08=f_eq07-6.0*(1.0/72)*rho*product;

        product=vx+vy-vz;
        f_eq09=(1.0/72)*rho*(1.0+3.0*product+4.5*product*product-square);
        f_eq10=f_eq09-6.0*(1.0/72)*rho*product;

        product=vx-vy+vz;
        f_eq11=(1.0/72)*rho*(1.0+3.0*product+4.5*product*product-square);
        f_eq12=f_eq11-6.0*(1.0/72)*rho*product;

        product=vx-vy-vz;
        f_eq13=(1.0/72)*rho*(1.0+3.0*product+4.5*product*product-square);
        f_eq14=f_eq13-6.0*(1.0/72)*rho*product;

        f00[destinationID] = ftemp_00 + (f_eq00-ftemp_00)*tau_inv;
        f01[destinationID] = ftemp_01 + (f_eq01-ftemp_01)*tau_inv;
        f02[destinationID] = ftemp_02 + (f_eq02-ftemp_02)*tau_inv;
        f03[destinationID] = ftemp_03 + (f_eq03-ftemp_03)*tau_inv;
        f04[destinationID] = ftemp_04 + (f_eq04-ftemp_04)*tau_inv;
        f05[destinationID] = ftemp_05 + (f_eq05-ftemp_05)*tau_inv;
        f06[destinationID] = ftemp_06 + (f_eq06-ftemp_06)*tau_inv;
        f07[destinationID] = ftemp_07 + (f_eq07-ftemp_07)*tau_inv;
        f08[destinationID] = ftemp_08 + (f_eq08-ftemp_08)*tau_inv;
        f09[destinationID] = ftemp_09 + (f_eq09-ftemp_09)*tau_inv;
        f10[destinationID] = ftemp_10 + (f_eq10-ftemp_10)*tau_inv;
        f11[destinationID] = ftemp_11 + (f_eq11-ftemp_11)*tau_inv;
        f12[destinationID] = ftemp_12 + (f_eq12-ftemp_12)*tau_inv;
        f13[destinationID] = ftemp_13 + (f_eq13-ftemp_13)*tau_inv;
        f14[destinationID] = ftemp_14 + (f_eq14-ftemp_14)*tau_inv;
    }
}
#endif //_CUDA_
