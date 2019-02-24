import numpy as np
import math
import matplotlib.pyplot as plt 
import cv2

def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    '''
    INPUT 
    im              the map 
    x_im,y_im       physical x,y positions of the grid map cells
    vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
    xs,ys           physical x,y,positions you want to evaluate "correlation" 

    OUTPUT 
    c               sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx] # 1 x 1076
            ix = np.int16(np.round((x1-xmin)/xresolution))
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr
def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
        (sx, sy)	start point of ray
        (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap 

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x,y))
class mapping:
    def __int__(self,maxx,minx,maxy,miny,res):
        self.maxx=maxx
        self.minx=minx
        self.maxy=maxy
        self.miny=miny
        self.res=res
        self.gam1=0.9
        self.gam0=0.1
        self.odds=np.log(gam1/gam0)
        self.shape=((int((self.maxx-self.minx)//self.res)+1),(int((self.maxy-self.miny)//self.res)+1))
        self.occupancymap=np.zeros(self.shape,dtype=np.unit8)
        self.logodds=np.zeros(self.shape,dtype=np.float64)
    
    def occmap(particle,lidar):
        w_t_b = np.array([[np.cos(particle[-1]),-np.sin(particle[-1])],[np.sin(particle[-1]),np.cos(particle[-1])]])
        coordinates = np.floor((np.dot(w_t_b,lidar)[:1,:] - np.array([self.minx,self.maxx])/self.res).astype(np.uint16)
        ends = coordinates.T.tolist()
        source = np.floor((np.array([particle[0],particle[1]]) - np.array([self.minx,self.maxx])/self.res).astype(np.uint16)
        for end in ends:
            scans = bresenham2D(source[0],source[1],end[0],end[1]).astype(np.uint16)
            self.logodds[scans[1][-1],scans[0][-1]] = self.logodds[scans[1][-1],scans[0][-1]] + self.odds
            self.logodds[scans[1][1:-1],scans[0][1:-1]] = self.logodds[scans[1][1:-1],scans[0][1:-1]] - self.odds
        self.logodds[scans[1][0],scans[0][0]] = self.logodds[scans[1][0],scans[0][0]] - self.odds
        P = 1/(1 + np.exp(-self.logodds))
        self.occupancy_map = (P>=0.9)*(1) + (P<0.05)*(-1)
        return self.occupancymap,self.logodds
                      
                          
                          
     def generate_texture(self,particle,texture,body_frame):
        wTb = np.array([[np.cos(particle[-1]),-np.sin(particle[-1])],[np.sin(particle[-1]),np.cos(particle[-1])],[0,0],[0,0])
        map_coordinates = np.floor((np.dot(wTb,body_frame)[:2,:] - self._grid_shift_vector.reshape(-1,1))/self._grid_stats["res"]).astype(np.uint16)#Transfer points into world co ordinates
        indices = np.logical_and(map_coordinates[1,:]<self.texture_map.shape[0],map_coordinates[0,:]<self.texture_map.shape[1])
        map_coordinates = map_coordinates[:,indices]
        self.texture_map[map_coordinates[1,:],map_coordinates[0,:],:] = texture/255
        P = 1/(1 + np.exp(-self.logodds))
        self.texture_map[:,:,0] = self.texture_map[:,:,0]*(P_occupied<0.05)
        self.texture_map[:,:,1] = self.texture_map[:,:,1]*(P_occupied<0.05)
        self.texture_map[:,:,2] = self.texture_map[:,:,2]*(P_occupied<0.05)
        return self.texture_map


from scipy.signal import butter, lfilter
class Lidardata:
    def __init__(self,minangle,maxangle,incrementangle,minrange,maxrange,ranges,timestamp,b_t_l):
        self.maxrange=maxrange
        self.minrange=minrange
        self.angles=np.arange(minangle,maxangle+ incrementangle,incrementangle,dtype=np.float64)[:1081]
        self.ranges = ranges
        self.timestamp = timestamp
        self.b_t_l=b_t_l

    
    def stateit(self,iteration):

        ranges = self.ranges[:,iteration]
        indValid = np.logical_and((ranges <= self.maxrange),(ranges >= self.minrange))
        
        ranges = ranges[indValid]
        angles = self.angles[indValid]

        x = ranges*np.cos(angles) + self.b_t_l[0]
        y = ranges*np.sin(angles) + self.b_t_l[1]
        pose=np.stack([x,y],axis=0)
        return pose

class IMUdata:
    def __init__(self,omega,timestamp):
        self.yawrate = omega[2,:]
        self.timestamp = timestamp
        fs = 1/(timestamp[1]-timestamp[0])
        fc = 10
        B, A = butter(1, fc / (fs / 2), btype='low') 
        self.yawrate = lfilter(B, A, self.yawrate, axis=0)



    def stateit(self,iteration):
        return self.yawrate[iteration]

class Encoderdata:
    def __init__(self,counts,timestamp):
        self.timestamps = timestamps
        left = 0.0022*(counts[0,:] + counts[2,:])/2
        right = 0.0022*(counts[1,:] + counts[3,:])/2
        self.displacement = (left + right)/2
        

    def stateit(self,iteration):
        return self.displacement[iteration]


class particlefilter:
    def __init__(self,N,threshold):

        self.particles = np.zeros((N,3))
        self.particle_weight = np.ones((N,1))/N
        self.threshold = threshold


    def prediction(self,displacement,angle_shift):
        noise = np.random.randn(self.particles.shape[0],self.particles.shape[1])*0.0001
        
        if angle_shift>=0:
            self.particles[:,0] = self.particles[:,0] + (displacement*np.sin(angle_shift/2)*(np.cos(self.particles[:,-1] + (angle_shift/2)))/(angle_shift/2)) 
            self.particles[:,1] = self.particles[:,1] + (displacement*np.sin(angle_shift/2)*(np.sin(self.particles[:,-1] + (angle_shift/2)))/(angle_shift/2)) 
            self.particles[:,2] = self.particles[:,2] + angle_shift
        self.particles = self.particles + noise
        

    def update(self,lidar,Obsmodel):
        occgrid = Obsmodel.occupancymap
        ph = np.zeros_like(self.particle_weight)
        x_im = np.arange(minx,maxy+res,res)
        y_im = np.arange(miny,maxy+res,res)

        for i in range(self.particles.shape[0]):
            x_range = np.arange(-0.2,0.2+0.05,0.05)
            y_range = np.arange(-0.2,0.2+0.05,0.05)
            corr = mapCorrelation(occgrid,x_im,y_im,lidar,x_range,y_range)
            ph[i] = np.max(corr)
            location = np.unravel_index(corr.argmax(), corr.shape)
            self.particles[i,0] = self.particles[i,0] + (location[0] - 4)*res
            self.particles[i,1] = self.particles[i,1] + (location[1] - 4)*res
        ph = np.exp(ph - np.max(ph))
        ph = ph/np.sum(ph)            
        
        self.particle_weight = ph*self.particle_weight
        self.particle_weight = self.particle_weight/np.sum(self.particle_weight)

    def sampling(self):
        j = 0
        c = self.particle_weight[0]
        new_particles = self.particles.copy()
        num_particles = self.particles.shape[0]
        for k in range(num_particles):
            u = np.random.uniform(0,1/num_particles)
            beta = u + (k/num_particles)
            while beta > c:
                j = j + 1
                c = c + self.particle_weight[j]
            new_particles[k,:] = self.particles[j,:]
        self.particles = new_particles
        self.particle_weight = np.ones((num_particles,1))/num_particles




