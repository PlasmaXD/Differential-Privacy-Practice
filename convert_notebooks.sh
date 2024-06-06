#!/bin/bash

# ファイルリスト
notebooks=(
    "data.ipynb"
    "dataanaytisc.ipynb"
    "recruit.ipynb"
    "recruit2.ipynb"
    "recruitML copy.ipynb"
    "recruitML ロジスティクス直し.ipynb"
    "recruitML.ipynb"
    "recruit記述統計へのDP適用.ipynb"
    "recruit記述統計バイアス補正 (コピー).ipynb"
    "recruit記述統計バイアス補正.ipynb"
    "sample.ipynb"
    "記述統計.ipynb"
)

# 各ノートブックをPDFに変換
for notebook in "${notebooks[@]}"; do
    jupyter nbconvert --to pdf "$notebook"
done
