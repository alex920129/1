
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

st.set_page_config(page_title="心脏病预测 · Debug 版", page_icon="🛠️", layout="centered")
st.title("🛠️ 心脏病预测（Debug 版）")

st.markdown("""
这个页面会 **自动在多处路径寻找模型**，并展示**当前工作目录与文件列表**，帮助定位“未找到可用模型”的原因。
""")

# ---------- 显示运行环境信息 ----------
cwd = Path.cwd()
here = Path(__file__).resolve().parent
st.write("**当前工作目录 (Path.cwd())**：", str(cwd))
st.write("**app 文件所在目录 (__file__)**：", str(here))

def list_dir(p: Path, depth=1):
    rows = []
    try:
        for item in p.iterdir():
            rows.append({"name": item.name, "is_dir": item.is_dir(), "size": (item.stat().st_size if item.is_file() else None)})
    except Exception as e:
        rows.append({"name": f"<无法列出: {e}>", "is_dir": "", "size": ""})
    return rows

with st.expander("查看当前目录文件列表", expanded=False):
    import pandas as pd
    st.dataframe(pd.DataFrame(list_dir(cwd)))

with st.expander("查看 app.py 所在目录文件列表", expanded=False):
    import pandas as pd
    st.dataframe(pd.DataFrame(list_dir(here)))

# ---------- 寻找模型/CSV ----------
candidate_dirs = [
    cwd,
    here,
    cwd / "model",
    cwd / "models",
    cwd / "data",
    cwd / "assets",
]

env_model = os.getenv("MODEL_PATH")
env_csv = os.getenv("CSV_PATH")

model_candidates = []
csv_candidates = []

if env_model:
    model_candidates.append(Path(env_model))
for d in candidate_dirs:
    model_candidates.append(d / "best_model.pkl")

if env_csv:
    csv_candidates.append(Path(env_csv))
for d in candidate_dirs:
    csv_candidates.append(d / "heart_disease_uci .csv")  # 保持原文件名（含空格）

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

MODEL_PATH = first_existing(model_candidates)
CSV_PATH = first_existing(csv_candidates)

st.write("**模型候选路径（按顺序）**：", [str(p) for p in model_candidates[:6]] + (["..."] if len(model_candidates) > 6 else []))
st.write("**找到的模型路径**：", str(MODEL_PATH) if MODEL_PATH else ":x: 没找到")

st.write("**CSV 候选路径（按顺序）**：", [str(p) for p in csv_candidates[:6]] + (["..."] if len(csv_candidates) > 6 else []))
st.write("**找到的 CSV 路径**：", str(CSV_PATH) if CSV_PATH else ":warning: 没找到（非致命）")

def is_git_lfs_pointer(path: Path):
    try:
        if path.is_file() and path.stat().st_size < 1024:
            head = path.read_text(errors="ignore")[:200]
            return "git-lfs.github.com/spec" in head
    except Exception:
        pass
    return False

if MODEL_PATH:
    if is_git_lfs_pointer(MODEL_PATH):
        st.error("检测到 **Git LFS 指针文件**，而非真实模型。请在仓库启用 Git LFS，并确保部署时 LFS 对象被正确拉取。")
        st.stop()

# ---------- 允许手动上传兜底 ----------
st.sidebar.header("📂 兜底上传")
model_file = None
if MODEL_PATH is None:
    model_file = st.sidebar.file_uploader("上传模型（.pkl）", type=["pkl"])
csv_file = None
if CSV_PATH is None:
    csv_file = st.sidebar.file_uploader("上传示例 CSV", type=["csv"])

# ---------- 加载模型/CSV ----------
def load_model(file_like, path: Path):
    try:
        if file_like is not None:
            return pickle.load(file_like)
        elif path is not None and path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"模型加载失败：{e}")
        st.stop()
    return None

def load_csv(file_like, path: Path):
    try:
        if file_like is not None:
            return pd.read_csv(file_like)
        elif path is not None and path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"CSV 读取失败：{e}")
        return None
    return None

model = load_model(model_file, MODEL_PATH)
df = load_csv(csv_file, CSV_PATH)

if model is None:
    st.error("仍然没有加载到模型。请确认文件名与路径，并检查是否为 Git LFS 指针。")
    st.stop()

# ---------- 推断特征名并生成表单 ----------
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
elif df is not None:
    feature_names = list(df.columns)
else:
    st.error("无法确定特征名。请提供 CSV 或使用带 feature_names_in_ 的模型（Pipeline 最佳）。")
    st.stop()

st.sidebar.header("🧮 填写特征值")

def build_default(series: pd.Series):
    if series is None or series.empty:
        return ""
    if pd.api.types.is_numeric_dtype(series):
        return float(series.median())
    try:
        return series.mode().iloc[0]
    except Exception:
        return ""

def widget_for_column(col: str):
    col_series = df[col] if (df is not None and col in df.columns) else None
    is_numeric = pd.api.types.is_numeric_dtype(col_series) if col_series is not None else True
    if is_numeric:
        if col_series is not None and not col_series.empty:
            q01 = float(col_series.quantile(0.01))
            q99 = float(col_series.quantile(0.99))
            default_val = build_default(col_series)
            non_na = col_series.dropna()
            is_int_like = len(non_na) > 0 and all(float(x).is_integer() for x in non_na[: min(200, len(non_na))])
            step = 1.0 if is_int_like else 0.1
        else:
            q01, q99, default_val, step = 0.0, 100.0, 0.0, 1.0
        return st.sidebar.number_input(f"{col}", min_value=q01, max_value=q99, value=float(default_val), step=step, key=f"num_{col}")
    else:
        uniques = col_series.dropna().unique().tolist() if (col_series is not None and not col_series.empty) else []
        if 0 < len(uniques) <= 15:
            default_val = build_default(col_series) if col_series is not None else (uniques[0] if uniques else "")
            idx = uniques.index(default_val) if default_val in uniques else 0
            return st.sidebar.selectbox(f"{col}", options=uniques or [""], index=idx, key=f"cat_{col}")
        else:
            default_val = build_default(col_series) if col_series is not None else ""
            return st.sidebar.text_input(f"{col}", value=str(default_val), key=f"text_{col}")

user_input = {col: widget_for_column(col) for col in feature_names}
X = pd.DataFrame([user_input], columns=feature_names)

st.markdown("### 你的输入")
st.dataframe(X, use_container_width=True)

if st.sidebar.button("预测", type="primary", use_container_width=True):
    try:
        y_pred = model.predict(X)
        st.markdown("### 预测结果")
        st.success(f"预测类别：**{y_pred[0]}**")
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:
                    st.info(f"阳性概率：**{proba[0,1]:.3f}**；阴性概率：{proba[0,0]:.3f}")
                else:
                    import pandas as pd
                    prob_series = pd.Series(proba[0], index=getattr(model, "classes_", list(range(proba.shape[1]))))
                    st.dataframe(prob_series.to_frame("probability"))
            except Exception:
                pass
    except Exception as e:
        st.error(f"预测失败：{e}")
        st.caption("多为列名/类型不一致或缺少预处理。建议将预处理与模型封装到 sklearn Pipeline 后再保存。")
