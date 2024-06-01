# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Define a class to receive the characteristics of each line detection
class Lane():
    def __init__(self):
        # 当前的图像
        self.current_warped_binary = None
        # 当前图片的尺寸
        self.current_warped_binary_shape = []

        # 检测到的车道线像素的横坐标 x values for detected line pixels
        self.allx = None

        # 检测到的车道线像素的纵坐标 y values for detected line pixels
        self.ally = None

        # 以纵坐标为自变量，取值空间
        self.ploty = None

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 是否检测到车道线 was the line detected in the last iteration?
        self.detected = False

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 保存的数据量
        self.n = 5

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 最近n个帧的拟合曲线 x values of the last n fits of the line
        self.recent_fitted_xs = []

        # 最近n个帧的平均拟合曲线 average x values of the fitted line over the last n iterations
        self.average_fitted_x = []

        # 当前帧的拟合曲线
        self.current_fitted_x = []

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 最近n个帧的拟合函数
        self.recent_fits = []

        # 最近n个帧的拟合函数 polynomial coefficients averaged over the last n iterations
        self.average_fit = []

        # 当前帧的拟合函数 polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # 拟合函数的误差 difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 半径 radius of curvature of the line in some units
        self.radius_of_curvature = []

        # 车辆在车道线之间距离 distance in meters of vehicle center from the line
        self.line_base_pos = None

    # 对全新的帧进行车道线像素检测
    def find_lane_pixels(self, binary_warped, location):
        self.current_warped_binary = binary_warped
        self.current_warped_binary_shape = binary_warped.shape
        self.ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result
        # out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)

        if location == "left":
            base = np.argmax(histogram[:midpoint])
        elif location == "right":
            base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 80
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()  # 扁平化后非零值点的列表
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows

        current = base

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []
        # right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_x_low = current - margin
            win_x_high = current + margin

            # # Draw the windows on the visualization image
            # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
            #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
            # cv2.rectangle(out_img, (win_xright_low, win_y_low),
            #               (win_xright_high, win_y_high), (0, 255, 0), 2)

            # 形成对每个像素的bool值
            # Identify the nonzero pixels in x and y within the window #
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        self.allx = x
        self.ally = y

        return x, y

    # 在之前的plot基础上找车道线
    def search_pixel_around_poly(self, binary_warped):

        self.current_warped_binary = binary_warped
        self.current_warped_binary_shape = binary_warped.shape
        self.ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 80

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        fit = self.recent_fits[-1]

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        lane_inds = ((nonzerox > (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] - margin)) & (
                    nonzerox < (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] + margin)))

        # Again, extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        self.allx = x
        self.ally = y

        return x, y

    def fit_polynomial(self):
        ploty = self.ploty

        # Fit a second order polynomial to each using `np.polyfit`
        fit = np.polyfit(self.ally, self.allx, 2)

        # 存储当前结果
        self.current_fit = fit

        # 计算误差
        if len(self.recent_fits) == 0:
            self.diffs = [0,0,0]
        else:
            new = np.array(self.current_fit)
            old = np.array(self.recent_fits[-1])
            self.diffs = new - old

        # 存储为历史结果
        if len(self.recent_fits) < self.n:
            self.recent_fits.append(self.current_fit)
        elif len(self.recent_fits) == self.n:
            self.recent_fits.pop(0)
            self.recent_fits.append(self.current_fit)
        else:
            self.recent_fits.append(self.current_fit)
            self.recent_fits = self.recent_fits[-self.n:]  # 后面n个

        # 计算当前平均
        self.average_fit = np.array(self.recent_fits).mean(axis=0)


        try:
            x_fitted = self.average_fit[0] * ploty ** 2 + self.average_fit[1] * ploty + self.average_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            x_fitted = 1 * ploty ** 2 + 1 * ploty
            self.detected = False
        else:
            self.detected = True

        self.current_fitted_x = x_fitted

        # 存储为历史结果
        if len(self.recent_fitted_xs) < self.n:
            self.recent_fitted_xs.append(self.current_fitted_x)
        elif len(self.recent_fitted_xs) == self.n:
            self.recent_fitted_xs.pop(0)
            self.recent_fitted_xs.append(self.current_fitted_x)
        else:
            self.recent_fitted_xs.append(self.current_fitted_x)
            self.recent_fitted_xs = self.recent_fitted_xs[-self.n:]  # 后面n个

        self.average_fitted_x = np.array(self.recent_fitted_xs).mean(axis=0)

        return self.average_fitted_x

    def fit(self, binary_warped,location,sequence=True):
        if sequence:
            if not self.detected:
                # 没有检测到，重新开始检测
                self.find_lane_pixels(binary_warped,location)
            else:
                # 从上一次周围开始检测
                self.search_pixel_around_poly(binary_warped)
                # TODO 如果两次检测的误差较大怎么办？
                # TODO 是否存在

            self.fit_polynomial()
            # if np.abs(self.diffs).sum() > 20:
            #     self.current_fit = np.array(self.recent_fits[:-1]).mean(axis=0)
            #     self.recent_fits[-1] = self.current_fit
            #     self.average_fit = np.array(self.recent_fits).mean(axis=0)
            #
            #     self.current_fitted_x = np.array(self.recent_fitted_xs[:-1]).mean(axis=0)
            #     self.recent_fitted_xs[-1] = self.current_fitted_x
            #     self.average_fitted_x = np.array(self.recent_fitted_xs).mean(axis=0)
        else:
            self.find_lane_pixels(binary_warped, location)
            self.fit_polynomial()

    def measure_curvature_real(self,ploty, x, y):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculation of R_curve (radius of curvature)
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

        self.radius_of_curvature = curverad

        return curverad


if __name__ == "__main__":
    from lane.perspective import perspective,src,dst
    from lane.gaussian_blur import gaussian_blur
    from lane.combined_threshold import combined_threshold
    from lane.measure_vehicle_pos import measure_vehicle_pos
    from lane.draw_lane import draw_lane

    image = mpimg.imread('../output_images/undistorted/straight_lines1-undistorted.jpg')

    image = gaussian_blur(image, 3)
    combined = combined_threshold(image, ksize=3,
                                  th=[[20, 100], [25, 254], [100, 250], [0.6, 1.2], [180, 254], [250, 0]])
    combined = gaussian_blur(combined, 3)
    perspectived_img = perspective(combined,src,dst)
    
    # plt.imshow(perspectived_img,cmap="gray")
    # plt.show()

    left_lane = Lane()
    left_lane.fit(perspectived_img,"left")

    right_lane = Lane()
    right_lane.fit(perspectived_img, "right")


    result = left_lane.visual(perspectived_img,"left")
    plt.imshow(result)
    result = right_lane.visual(perspectived_img, "right")
    plt.imshow(result)
    plt.show()
    # # 计算曲率
    # left_r = left_lane.measure_curvature_real(left_lane.ploty, left_lane.average_fitted_x, left_lane.ploty)
    # right_r = left_lane.measure_curvature_real(right_lane.ploty, right_lane.average_fitted_x, right_lane.ploty)
    #
    # # 计算偏移值
    # v = measure_vehicle_pos(left_lane.average_fitted_x, right_lane.average_fitted_x,left_lane.current_warped_binary_shape[1])
    #
    # # 绘制车道线
    # img = draw_lane(image, combined, dst, src,left_lane.current_fitted_x, right_lane.current_fitted_x, right_lane.ploty)


    # plt.imshow(img)


    # # 打印文字
    # plt.text(0,60,"Radius of Curvature = %d(m)"%int(r),fontdict={'size': 20, 'color': 'w'})
    # plt.text(0,120, "Vehicle is %.2f(m) left of center" % v, fontdict={'size': 20, 'color': 'w'})
    # plt.show()








