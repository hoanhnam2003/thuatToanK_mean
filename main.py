import math
import random


# Bước 1: Tải dữ liệu Iris từ tệp local
def load_iris_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    dataset = []
    labels = []

    label_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

    for line in lines:
        values = line.strip().split(',')
        if values[-1] in label_mapping:
            dataset.append([float(x) for x in values[:-1]])
            labels.append(label_mapping[values[-1]])

    return dataset, labels


# Bước 2: K-means với lựa chọn các cụm ngẫu nhiên
def kmeans(data, k, max_iters=100):
    centroids = random.sample(data, k)
    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]

        for point in data:
            distances = [sum((point[i] - centroid[i]) ** 2 for i in range(len(point))) ** 0.5 for centroid in centroids]
            clusters[distances.index(min(distances))].append(point)

        for i in range(k):
            if clusters[i]:
                centroids[i] = [sum(dim) / len(clusters[i]) for dim in zip(*clusters[i])]

    return clusters, centroids


# Bước 3: Tính toán các chỉ số F1-score, RAND Index, NMI và Davies-Bouldin Index
def f1_score(predicted, true_labels):
    tp = sum(1 for i in range(len(predicted)) if predicted[i] == true_labels[i])
    fp = sum(1 for i in range(len(predicted)) if predicted[i] != true_labels[i] and true_labels[i] == 1)
    fn = sum(1 for i in range(len(predicted)) if predicted[i] != true_labels[i] and true_labels[i] == 0)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0


def rand_index(predicted, true_labels):
    a = b = c = d = 0
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            if (predicted[i] == predicted[j]) == (true_labels[i] == true_labels[j]):
                a += 1
            else:
                b += 1
    return a / (a + b)


def mutual_information(predicted, true_labels):
    def entropy(labels):
        total = len(labels)
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return -sum((count / total) * math.log(count / total, 2) for count in counts.values())

    h_true = entropy(true_labels)
    h_pred = entropy(predicted)

    joint_dist = {}
    for i in range(len(predicted)):
        joint_dist[(predicted[i], true_labels[i])] = joint_dist.get((predicted[i], true_labels[i]), 0) + 1

    total = len(predicted)
    mi = sum((count / total) * math.log(total * count / (sum(1 for p in predicted if p == x) * sum(1 for t in true_labels if t == y)), 2)
             for (x, y), count in joint_dist.items())

    return mi / (h_true + h_pred) if h_true + h_pred > 0 else 0


def davies_bouldin_index(data, clusters, centroids):
    def euclidean_distance(p1, p2):
        return sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))) ** 0.5

    db_index = 0
    for i, cluster_i in enumerate(clusters):
        s_i = sum(euclidean_distance(point, centroids[i]) for point in cluster_i) / len(cluster_i)
        max_ratio = 0
        for j, cluster_j in enumerate(clusters):
            if i != j:
                s_j = sum(euclidean_distance(point, centroids[j]) for point in cluster_j) / len(cluster_j)
                d_ij = euclidean_distance(centroids[i], centroids[j])
                max_ratio = max(max_ratio, (s_i + s_j) / d_ij if d_ij > 0 else 0)
        db_index += max_ratio

    return db_index / len(clusters) if clusters else 0


# Chương trình chính
filepath = r"C:\Users\Admin\Downloads\iris\iris.data"
data, labels = load_iris_data(filepath)

k = 3
clusters, centroids = kmeans(data, k)

predicted_labels = []
for point in data:
    distances = [sum((point[i] - centroid[i]) ** 2 for i in range(len(point))) ** 0.5 for centroid in centroids]
    predicted_labels.append(distances.index(min(distances)))

f1 = f1_score(predicted_labels, labels)
rand_idx = rand_index(predicted_labels, labels)
nmi = mutual_information(predicted_labels, labels)
db_index = davies_bouldin_index(data, clusters, centroids)

print("F1-score:", f1)
print("RAND Index:", rand_idx)
print("NMI:", nmi)
print("Davies-Bouldin Index:", db_index)
