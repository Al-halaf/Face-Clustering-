import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.ndimage import sobel, zoom


def compute_grad(image):
    grad_x = sobel(image, axis=0)
    grad_y = sobel(image, axis=1)
    gradient = np.hypot(grad_x, grad_y)

    return gradient


def compute_orientation(image):
    grad_x = sobel(image, axis=0)
    grad_y = sobel(image, axis=1)

    orientation = np.arctan2(grad_y, grad_x)

    return orientation


def downsampling(faces, num_faces, height, width, downsampling_factor=2):
    downsampled_height = height // downsampling_factor
    downsampled_width = width // downsampling_factor

    # Downsample the faces
    downsampled_faces = np.zeros((downsampled_width * downsampled_height, num_faces))
    for i in range(num_faces):
        face = faces[:, i].reshape(height, width)
        downsample = zoom(face, 1 / downsampling_factor)
        downsampled_faces[:, i] = downsample.reshape(downsampled_height * downsampled_width)

    return downsampled_faces


def gradient_affinity(faces, height, width, sigma=1000):
    # Get the number of faces in the dataset
    num_faces = faces.shape[1]

    # Compute the gradients and orientation of the images
    gradients = np.zeros((width * height, num_faces))
    orientations = np.zeros((width * height, num_faces))
    for i in range(num_faces):
        face = faces[:, i].reshape(height, width) / (
            np.linalg.norm(faces[:, i].reshape(height, width), ord=2))

        grad = compute_grad(face)
        orientation = compute_orientation(face)
        orientations[:, i] = orientation.reshape(height * width)
        gradients[:, i] = grad.reshape(height * width)

    # Normalise the orientations from {-pi, pi} to {0, 1}
    orientations = (orientations + np.pi) / (2 * np.pi)

    # Compute the affinity matrix A
    O = np.zeros((num_faces, num_faces))
    for i in range(num_faces):
        O[i, :] = np.sum((orientations[:, i, np.newaxis] - orientations) ** 2, axis=0)

    # Compute the M matrix (pairwise gradient norm differences)
    norm_gradients = np.linalg.norm(gradients, axis=0)
    M = (norm_gradients[:, np.newaxis] - norm_gradients) ** 2

    A = np.exp((-1 / (2 * sigma ** 2)) * (M + O))

    return A


def main():
    # Load the data set
    mat_contents = loadmat("allFaces.mat")
    faces = mat_contents["faces"]
    nfaces = mat_contents["nfaces"]
    num_faces = np.sum(nfaces)
    height = int(mat_contents["m"][0][0])
    width = int(mat_contents["n"][0][0])
    factor = 2

    # Label the images
    labels = list()
    for i in range(nfaces.shape[1]):
        for j in range(nfaces.item(i)):
            labels.append(i)

    # Plot four of the faces in the dataset
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(faces[:, 0].reshape(height, width).T, cmap="grey")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(faces[:, nfaces.item(0)].reshape(height, width).T, cmap="grey")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(faces[:, np.sum(nfaces[:, :2])].reshape(height, width).T, cmap="grey")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(faces[:, np.sum(nfaces[:, :3])].reshape(height, width).T, cmap="grey")
    axs[1, 1].axis("off")

    fig.suptitle("The first four faces in the dataset", fontsize=16)
    plt.show()

    # Plot the same face under different lighting conditions
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(faces[:, 0].reshape(height, width).T, cmap="grey")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(faces[:, 25].reshape(height, width).T, cmap="grey")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(faces[:, 30].reshape(height, width).T, cmap="grey")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(faces[:, 60].reshape(height, width).T, cmap="grey")
    axs[1, 1].axis("off")

    fig.suptitle("The first fce under different lighting conditions", fontsize=16)
    plt.show()

    # Clustering using Kmeans
    # Kmeans clustering accuracy: 1.17%
    Km = KMeans(n_clusters=nfaces.shape[1], random_state=0)
    Km_clusters = Km.fit_predict(faces.T)

    Km_acc = adjusted_rand_score(labels_true=labels, labels_pred=Km_clusters)
    print(f"K-means clustering accuracy: {Km_acc * 100:.2f}%")

    # Downsample the faces for faster clustering
    downsampled_faces = downsampling(faces, num_faces, height, width, downsampling_factor=factor)

    # Cluster the faces using gradient affinity
    # Clustering using gradient affinity accuracy: 85.59%
    A_grad = gradient_affinity(downsampled_faces, height=height//factor, width=width//factor)

    spec = SpectralClustering(n_clusters=nfaces.shape[1], affinity="precomputed", random_state=0)
    grad_clusters = spec.fit_predict(A_grad)

    # Check the accuracy score
    grad_acc = adjusted_rand_score(labels_true=labels, labels_pred=grad_clusters)
    print(f"Clustering using gradient affinity accuracy: {grad_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
