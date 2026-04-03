import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import graphviz
from IPython.display import display


print("--- BƯỚC 1: LOAD DỮ LIỆU HEALTHCARE (DIABETES) TỪ URL ---")

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)


selected_features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
X = df[selected_features].values
y = df['Outcome'].values # 0: Không tiểu đường, 1: Có tiểu đường

print("EDA Cơ bản (5 dòng đầu):")
display(df[selected_features + ['Outcome']].head())
print(f"\nKích thước dữ liệu gốc: {df.shape}")
print(f"Kích thước tập X (chỉ lấy 4 features): {X.shape}\n")

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Số mẫu tập Train: {X_train.shape[0]}, Số mẫu tập Test: {X_test.shape[0]}")

# CÀI ĐẶT DECISION TREE THỦ CÔNG

class Node:
    """Đại diện cho 1 nút trong cây quyết định"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       # Index của feature được chọn để chia
        self.threshold = threshold   # Ngưỡng chia
        self.left = left             # Nhánh con bên trái (Node)
        self.right = right           # Nhánh con bên phải (Node)
        self.value = value           # Giá trị dự đoán (chỉ có ở Node lá)

    def is_leaf_node(self):
        return self.value is not None


class ManualDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, criterion='gini'):
        self.min_samples_split = min_samples_split # Regularization: Số mẫu tối thiểu để chia
        self.max_depth = max_depth                 # Regularization: Độ sâu tối đa
        self.criterion = criterion                 # tiêu chí đo 'gini' hoặc 'entropy'
        self.root = None

    # --- BƯỚC 4: XÂY DỰNG CÂY ĐỆ QUY ---
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Điều kiện dừng (Dừng đệ quy khi đạt max_depth, node thuần nhất, hoặc ít mẫu)
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Tìm best split
        best_feature, best_threshold = self._best_split(X, y, n_features)

        # Nếu không tìm được cách chia nào tốt hơn thì tạo node lá
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Tách mảng và gọi đệ quy cho nhánh trái, phải
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    # --- BƯỚC 3: TÌM BEST SPLIT (GREEDY SEARCH) ---
    def _best_split(self, X, y, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column) # Thử tất cả các giá trị unique làm ngưỡng

            for thr in thresholds:
                gain = self._calculate_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _calculate_gain(self, y, X_column, threshold):
        impurity_func = self._gini if self.criterion == 'gini' else self._entropy
        parent_impurity = impurity_func(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        child_impurity = (len(left_idxs) / n) * impurity_func(y[left_idxs]) + \
                         (len(right_idxs) / n) * impurity_func(y[right_idxs])

        return parent_impurity - child_impurity

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    # --- BƯỚC 2: CÀI ĐẶT GINI / ENTROPY ---
    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum(ps ** 2)

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    # --- BƯỚC 5: HÀM PREDICT ---
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# Cell 4: BƯỚC 6 - Huấn luyện và so sánh mô hình thủ công với Sklearn
# Khởi tạo tham số Regularization
MAX_DEPTH = 3
MIN_SAMPLES_SPLIT = 10
CRITERION = 'gini'

print(f"--- BƯỚC 6: HUẤN LUYỆN VÀ SO SÁNH ---")
print(f"Cấu hình: max_depth={MAX_DEPTH}, min_samples_split={MIN_SAMPLES_SPLIT}, criterion='{CRITERION}'\n")

# 1. Huấn luyện mô hình thủ công
manual_clf = ManualDecisionTree(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, criterion=CRITERION)
manual_clf.fit(X_train, y_train)
manual_pred = manual_clf.predict(X_test)
manual_acc = accuracy_score(y_test, manual_pred)

# 2. Huấn luyện mô hình của Sklearn
sklearn_clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, criterion=CRITERION, random_state=42)
sklearn_clf.fit(X_train, y_train)
sklearn_pred = sklearn_clf.predict(X_test)
sklearn_acc = accuracy_score(y_test, sklearn_pred)

# Bảng so sánh
print(f"{'Mô hình':<30} | {'Accuracy trên tập Test':<20}")
print("-" * 55)
print(f"{'Manual Decision Tree (Nhóm làm)':<30} | {manual_acc:.4f}")
print(f"{'Sklearn Decision Tree':<30} | {sklearn_acc:.4f}\n")

# Cell 5: BƯỚC 7 - Visualize cây quyết định thủ công
class Visualizer:
    @staticmethod
    def print_tree(tree_model, feature_names=None, node=None, indent=""):
        if node is None:
            node = tree_model.root

        if node.is_leaf_node():
            class_name = "Có bệnh" if node.value == 1 else "Không bệnh"
            print(f"{indent}└── Predict: {class_name}")
            return

        feat_name = feature_names[node.feature] if feature_names else f"Feature {node.feature}"
        print(f"{indent}├── IF {feat_name} <= {node.threshold:.2f}:")
        Visualizer.print_tree(tree_model, feature_names, node.left, indent + "│   ")
        print(f"{indent}└── ELSE ({feat_name} > {node.threshold:.2f}):")
        Visualizer.print_tree(tree_model, feature_names, node.right, indent + "    ")

    @staticmethod
    def export_graphviz(tree_model, feature_names=None, class_names=None):
        dot = graphviz.Digraph(node_attr={'shape': 'box', 'fontname': 'Helvetica'})

        def add_nodes_edges(node, dot):
            node_id = str(id(node))
            if node.is_leaf_node():
                label = f"Dự đoán:\n{class_names[node.value]}"
                # Màu đỏ nhạt cho Có bệnh, xanh nhạt cho Không bệnh
                color = "#ffcccc" if node.value == 1 else "#ccffcc"
                dot.node(node_id, label, style="filled", fillcolor=color)
            else:
                feat_name = feature_names[node.feature] if feature_names else f"Feature {node.feature}"
                label = f"{feat_name} <= {node.threshold:.2f}"
                dot.node(node_id, label, style="filled", fillcolor="#e6f2ff")

                if node.left:
                    left_id = add_nodes_edges(node.left, dot)
                    dot.edge(node_id, left_id, label="Đúng (True)")
                if node.right:
                    right_id = add_nodes_edges(node.right, dot)
                    dot.edge(node_id, right_id, label="Sai (False)")
            return node_id

        if tree_model.root is not None:
            add_nodes_edges(tree_model.root, dot)
        return dot

print("1. VISUALIZE BẰNG TEXT:\n")
Visualizer.print_tree(manual_clf, feature_names=selected_features)

print("\n2. VISUALIZE BẰNG GRAPHVIZ (SƠ ĐỒ):")
target_names = ["Không bệnh", "Có bệnh"]
dot_data = Visualizer.export_graphviz(manual_clf, feature_names=selected_features, class_names=target_names)
display(dot_data) # Hiển thị trực tiếp đồ thị trên Colab
