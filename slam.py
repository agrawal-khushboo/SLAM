
import numpy as np 
import matplotlib.pyplot as plt
from slam_functions import*
class slam:
    def __init__(self,PF,Obs):
        self.PF = PF
        self.Obs = Obs

        
    def func(self,Lidardar,IMUdata,Encoderdata,Kinect=None):
        lidar = Lidardata.timestamp
        imu = IMU.timestamp
        encoder = Encoderdata.timestamp
        if Kinect!=None:
            cam,Ir = Kinect
            cam_tracking = cam.timestamp
            ir_tracking = Ir.timestamps
            texture_map = None

        x = []
        y = []

        for i in range(lidar.shape[0]):
            if i==0:
                scan = lidar[0]
                state = self.PF.most_likely
                occ =  self.Obs.occmap(state,scan)
            else:
                scan = lidar[i]
                imuiteration = np.logical_and(imu>=lidar[i-1],imu<lidar[i])

                if np.sum(imuiteration)==0:
                    angle=0
                else:
                    angle = np.mean(IMU[imuiteration])*(lidar[i] - lidar[i-1])

                encode = np.logical_and(encoder>=lidar[i-1],encoder<lidar[i])
                displacement = np.sum(encoder[encode])

                if displacement!=0 or angle!=0:
                    self.PF.prediction(displacement,angle)
                    self.PF.update(scan,self.Obs)
                
                
#                 occ,_ =  self.Obs.occmap(state,scan)

                if Kinect!=None:
                    cam_iteration = np.logical_and(cam_tracking>=lidar[i-1],cam_tracking<lidar[i]) 
                    ir_iteration = np.logical_and(ir_tracking>=lidar[i-1],ir_tracking<lidar[i]) 
                    if np.sum(cam_iteration)==1 and np.sum(ir_iteration)==1:
                        cam_iteration = np.argmax(cam_iteration)
                        ir_iteration = np.argmax(ir_iteration)
                        image = cam[cam_iteration]
                        rgbi,rgbj,body_frame = Ir[ir_iteration]
#                         texture_pixels = image[rgbj,rgbi]
                        texture_map = self.Obs.generate_texture(state,texture_pixels,body_frame)
                        plt.scatter(x,y,s=0.02,c='r')
                        plt.imshow(texture_map)
                        plt.savefig("Plots/{}.png".format(i))
                        plt.close()
  

            if i%10==0 and Kinect==None:
                plt.scatter(x,y,s=0.01,c='r')
                plt.imshow(occ,cmap='binary')
#                 plt.savefig("Plots/{}.png".format(i))
                plt.close()

                

       

if __name__ == '__main__':
    dataset = 23

    with np.load("Encoders%d.npz"%dataset) as data:
        count = data["counts"] # 4 x n encoder counts
        timestamp = data["timestamp"] # encoder time stamps
        encoder_reader = Encoder(count,timestamp)
    with np.load("Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
        lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
        lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
        lidar_range_min = data["range_min"] # minimum range value [m]
        lidar_range_max = data["range_max"] # maximum range value [m]
        lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
        lidar_stamps = data["timestamp"]  # acquisition times of the lidar scans
        translation = np.array([0,.150,0]) #relative shift of LIDAR wrt center of X
        robotdata = LIDAR(lidar_angle_min,lidar_angle_max,lidar_angle_increment,lidar_range_min,lidar_range_max,lidar_ranges,lidar_stamps,translation)

    with np.load("Imu%d.npz"%dataset) as data:
        imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
        imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
        imu_stamps = data["timestamp"]  # acquisition times of the imu measurements
        imu_reader = IMU(imu_angular_velocity,imu_stamps)
    try:
        with np.load("Kinect%d.npz"%dataset) as data:
            disp_stamps = data["disparity_timestamp"] # acquisition times of the disparity images
            rgb_stamps = data["rgb_timestamp"] # acquisition times of the rgb images
            RGB_prefix = "dataRGBD/RGB{0}/rgb{1}_".format(dataset,dataset)
            RGB_folder = "dataRGBD/RGB{0}/".format(dataset)
            disp_prefix = "dataRGBD/Disparity{0}/disparity{1}_".format(dataset,dataset)
            disp_folder = "dataRGBD/Disparity{0}/".format(dataset)
            cam = cam(RGB_folder,RGB_prefix,rgb_stamps)
            ir = IR(disp_folder,disp_prefix,disp_stamps)
            Kinect = (cam,ir)


    Obs = mapping(20,-20,20,-20,0.05)
    PF = particlefilter(100,1)
    X = slam(PF,Obs)
    X(robotdata,imu_reader,encoder_reader,Kinect)


