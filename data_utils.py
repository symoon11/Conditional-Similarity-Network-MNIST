import numpy as np
from PIL import Image

def change_color(train_data):
    num, size = train_data.shape
    output = np.zeros([num, size, 3], dtype=np.uint8)
    color = np.zeros([num, 3], dtype=np.uint8)
    for i in range(num):
        print(i)
        r = np.random.randint(0, 200)
        g = np.random.randint(0, 200)
        b = np.random.randint(0, 200)
        color[i][0] = r
        color[i][1] = g
        color[i][2] = b
        for j in range(size):
            gray_scale = train_data[i][j]
            if gray_scale!=0:
                output[i][j][0] = r*gray_scale
                output[i][j][1] = g*gray_scale
                output[i][j][2] = b*gray_scale

    return output, color

class DataShuffler(object):
    def __init__(self, data, label_color, label_digit, perc_train=0.5, perc_val=0.3, perc_test = 0.2):

        total_samples = data.shape[0]
        indexes = np.array(range(total_samples))
        np.random.shuffle(indexes)

        train_samples = int(round(total_samples * perc_train))
        validation_samples = int(round(total_samples * perc_val))

        self.train_data = data[indexes[0:train_samples], :, :, :]
        self.train_label_digit = label_digit[indexes[0:train_samples]]
        self.train_label_color = label_color[indexes[0:train_samples], :]

        self.validation_data = data[indexes[train_samples:train_samples+validation_samples], :, :, :]
        self.validation_label_digit = label_digit[indexes[train_samples:train_samples+validation_samples]]
        self.validation_label_color = label_color[indexes[train_samples:train_samples+validation_samples], :]

        self.test_data = data[indexes[train_samples+validation_samples:], :, :, :]
        self.test_label_digit = label_digit[indexes[train_samples+validation_samples:]]
        self.test_label_color = label_color[indexes[train_samples+validation_samples:], :]

    def save_image(self, feat, i, index):
        img = Image.fromarray(self.test_data[index, :, :, :])
        file_name = "public/"+feat+"_"+str(i)+".jpg"
        img.save(file_name)

    def get_triplet(self, n_triplets, feature="color", training=True):

        def get_one_triplet_digit(input_data, input_label_digit):

            label_positive, label_negative = np.random.choice(10, 2, replace=False)

            indexes = np.where(input_label_digit == label_positive)[0]
            np.random.shuffle(indexes)

            data_anchor = input_data[indexes[0], :, :, :]
            data_positive = input_data[indexes[1], :, :, :]

            indexes = np.where(input_label_digit == label_negative)[0]
            np.random.shuffle(indexes)
            data_negative = input_data[indexes[0], :, :, :]

            return data_anchor, data_positive, data_negative

        def get_one_triplet_color(input_data, input_label_color):

            r = np.random.randint(0, 200)
            g = np.random.randint(0, 200)
            b = np.random.randint(0, 200)
            label_anchor = [r, g, b]
            label_anchor = np.reshape(label_anchor, [1, 3])

            i = 5
            while True:
                indexes = np.where(np.sum(np.square(input_label_color - label_anchor), axis=1) < 100 * i)[0]
                if indexes.size > 1:
                    break
                i += 1

            np.random.shuffle(indexes)

            data_anchor = input_data[indexes[0], :, :, :]
            data_positive = input_data[indexes[1], :, :, :]

            i = 50
            while True:
                indexes = np.where(np.sum(np.square(input_label_color - label_anchor), axis=1) > 100 * i)[0]
                if indexes.size > 0:
                    break
                i -= 1

            np.random.shuffle(indexes)
            data_negative = input_data[indexes[0], :, :, :]

            return data_anchor, data_positive, data_negative

        c = self.train_data.shape[3]
        w = self.train_data.shape[1]
        h = self.train_data.shape[2]

        data_a = np.zeros(shape=(n_triplets, w, h, c), dtype='float32')
        data_p = np.zeros(shape=(n_triplets, w, h, c), dtype='float32')
        data_n = np.zeros(shape=(n_triplets, w, h, c), dtype='float32')

        if feature == "color":
            for i in range(n_triplets):
                data_a[i], data_p[i], data_n[i] = get_one_triplet_color(self.train_data, self.train_label_color)
        else:
            for i in range(n_triplets):
                data_a[i], data_p[i], data_n[i] = get_one_triplet_digit(self.train_data, self.train_label_digit)

        return data_a, data_p, data_n