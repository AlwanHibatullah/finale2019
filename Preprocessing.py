import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import sys

np.set_printoptions(threshold=sys.maxsize)

class Preprocessing:

    def showimages(self, src_img, grey_img, final_thr):
        cv2.namedWindow('Hasil Segmentasi', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Citra Threshold', cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow('Binary Image', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Citra Grayscale', cv2.WINDOW_AUTOSIZE)

        cv2.imshow("Hasil Segmentasi", src_img)
        # cv2.imshow("Binary Image", bin_img)
        cv2.imshow("Citra Grayscale", grey_img)
        cv2.imshow("Citra Threshold", final_thr)


    def line_array(self, array):
        list_x_upper = []
        list_x_lower = []
        for y in range(5, len(array) - 5):
            s_a, s_p = self.strtline(y, array)
            e_a, e_p = self.endline(y, array)
            if s_a >= 7 and s_p >= 5:
                list_x_upper.append(y)
            # bin_img[y][:] = 255
            if e_a >= 5 and e_p >= 7:
                list_x_lower.append(y)
        # bin_img[y][:] = 255
        return list_x_upper, list_x_lower

    def strtline(self, y, array):
        count_ahead = 0
        count_prev = 0
        for i in array[y:y + 10]:
            if i > 3:
                count_ahead += 1
        for i in array[y - 10:y]:
            if i == 0:
                count_prev += 1
        return count_ahead, count_prev

    def endline(self, y, array):
        count_ahead = 0
        count_prev = 0
        for i in array[y:y + 10]:
            if i == 0:
                count_ahead += 1
        for i in array[y - 10:y]:
            if i > 3:
                count_prev += 1
        return count_ahead, count_prev

    def endline_word(self, y, array, a):
        count_ahead = 0
        count_prev = 0
        for i in array[y:y + 2 * a]:
            if i < 2:
                count_ahead += 1
        for i in array[y - a:y]:
            if i > 2:
                count_prev += 1
        return count_prev, count_ahead

    def end_line_array(self, array, a):
        list_endlines = []
        for y in range(len(array)):
            e_p, e_a = self.endline_word(y, array, a)
            # print(e_p, e_a)
            if e_a >= int(1.5 * a) and e_p >= int(0.7 * a):
                list_endlines.append(y)
        return list_endlines

    def refine_endword(self, array):
        refine_list = []
        for y in range(len(array) - 1):
            if array[y] + 1 < array[y + 1]:
                refine_list.append(array[y])
        # refine_list.append(array[-1])
        return refine_list

    def refine_array(self, array_upper, array_lower):
        upperlines = []
        lowerlines = []
        for y in range(len(array_upper) - 1):
            if array_upper[y] + 5 < array_upper[y + 1]:
                upperlines.append(array_upper[y] - 10)
        for y in range(len(array_lower) - 1):
            if array_lower[y] + 5 < array_lower[y + 1]:
                lowerlines.append(array_lower[y] + 10)

        upperlines.append(array_upper[-1] - 10)
        lowerlines.append(array_lower[-1] + 10)

        return upperlines, lowerlines

    def letter_width(self, contours):
        letter_width_sum = 0
        count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                x, y, w, h = cv2.boundingRect(cnt)
                letter_width_sum += w
                count += 1

        return letter_width_sum / count

    def end_wrd_dtct(self, width, lines, i, bin_img, mean_lttr_width):
        count_y = np.zeros(shape=width)
        for x in range(width):
            for y in range(lines[i][0], lines[i][1]):
                if bin_img[y][x] == 255:
                    count_y[x] += 1
        end_lines = self.end_line_array(count_y, int(mean_lttr_width))
        # print(end_lines)
        endlines = self.refine_endword(end_lines)
        for x in endlines:
            self.final_thr[lines[i][0]:lines[i][1], x] = 255
        return endlines

    def letter_seg(self, lines_img, x_lines, i):
        copy_img = lines_img[i].copy()
        x_linescopy = x_lines[i].copy()

        # letter_img = []
        letter_k = []

        contours, hierarchy = cv2.findContours(copy_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                # letter_img.append(lines_img[i][y:y+h, x:x+w])
                letter_k.append((x, y, w, h))

        letter = sorted(letter_k, key=lambda student: student[0])
        # print(len(letter))

        word = 1
        letter_index = 0
        for e in range(len(letter)):
            if (letter[e][0] < x_linescopy[0]):
                letter_index += 1
                letter_img_tmp = lines_img[i][letter[e][1] - 5:letter[e][1] + letter[e][3] + 5,
                                 letter[e][0] - 5:letter[e][0] + letter[e][2] + 5]
                letter_img = cv2.resize(letter_img_tmp, dsize=(64,64), interpolation=cv2.INTER_AREA)
                cv2.imwrite('./__temp__/' + str(i + 1) + '_' + str(word) + '_' + str(letter_index) + '.jpg',
                            255 - letter_img)
            else:
                x_linescopy.pop(0)
                word += 1
                letter_index = 1
                letter_img_tmp = lines_img[i][letter[e][1] - 5:letter[e][1] + letter[e][3] + 5,
                                 letter[e][0] - 5:letter[e][0] + letter[e][2] + 5]
                letter_img = cv2.resize(letter_img_tmp, dsize=(64,64), interpolation=cv2.INTER_AREA)
                cv2.imwrite('./__temp__/' + str(i + 1) + '_' + str(word) + '_' + str(letter_index) + '.jpg',
                            255 - letter_img)
        # print(letter[e][0],x_linescopy[0], word)

    def countFile(self, folderdest):
        path = './data_training/'+folderdest
        files = len(next(os.walk(path))[2])
        return files

    def moveFiles(self, folderdest):
        os.chdir('..')

        # print(os.getcwd())

        source = './__temp__/'
        dest = './data_training/'+folderdest

        file = os.listdir(source)

        for f in file:
            shutil.move(source+f, dest)

    def renameFile(self, folderdest):
        # i=1

        # Check Number of Files Inside Folder Destination
        num = self.countFile(folderdest)
        print(num)
        if num == 0:
            i=1
        else:
            i=num+1

        # Rename All Files Inside Source Folder
        os.chdir('./__temp__')

        for filename in os.listdir():
            dst = str(i) + ".jpg"
            src = filename

            os.rename(src, dst)
            i += 1

        # Move All Files from Source to Destination Folder
        self.moveFiles(folderdest)

    # def moveTest(self, folderdest):
    #     os.chdir('..')
    #
    #     # print(os.getcwd())
    #
    #     source = './__temp__/'
    #     dest = './data_training/'+folderdest
    #
    #     file = os.listdir(source)
    #
    #     for f in file:
    #         shutil.move(source+f, dest)

    def renameTest(self):
        i=1

        # Check if there is files inside data_testing folder, if exist then remove
        folder = './data_testing/test'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path) : shutil.rmtree(file_path)
            except Exception as e:
                print(e)

        # Rename All Files Inside Source Folder
        os.chdir('./__temp__')

        for filename in os.listdir():
            dst = str(i) + ".jpg"
            src = filename

            os.rename(src, dst)
            i += 1

        # Move All Files from Source to Destination Folder

        # self.moveFiles(folderdest)
        os.chdir('..')
        source = './__temp__/'
        dest = './data_testing/test/'
        file = os.listdir(source)

        for f in file:
            shutil.move(source+f, dest)