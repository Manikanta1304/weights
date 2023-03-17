# def img_generator(img_list):
#     # TODO: here
#       yield # ...


def img_generator(img_list):
    for img_path in img_list:
        img = Image.open(os.path.join('Images', img_path))
        img = img.resize((299, 299))
        img = np.array(img)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        yield img


# session was getting crashed while encoding for all the images, hence taking a sample
train_list, test_list = train_df['image'].head(2000).to_list(), test_df['image'].head(500).to_list()
num_train_images, num_test_images = len(train_list), len(test_list)


train_matrix = np.zeros((num_train_images, 299, 299, 3))
test_matrix = np.zeros((num_test_images, 299, 299, 3))

generator = img_generator(train_list)
for i in range(num_train_images):
    train_matrix[i, :, :, :] = next(generator)

generator = img_generator(test_list)
for i in range(num_test_images):
    test_matrix[i, :, :, :] = next(generator)
