#!/bin/bash

# LaTeX编译脚本 - 编译 main.tex
# 使用pdflatex进行编译，按需处理bibtex

echo "开始编译 main.tex ..."

# 检查 main.tex 文件是否存在
if [ ! -f "main.tex" ]; then
    echo "错误：main.tex 文件不存在！"
    exit 1
fi

# 第一次编译 - 生成.aux文件
echo "第一次 pdflatex 编译..."
pdflatex -shell-escape -interaction=nonstopmode main.tex >/dev/null

# 如存在 .bib 文件则运行 bibtex（假定主文件名为 main）
if ls *.bib >/dev/null 2>&1; then
    echo "检测到 .bib 文件，运行 bibtex ..."
    bibtex main >/dev/null || true
fi

# 第二次编译 - 处理交叉引用
echo "第二次 pdflatex 编译..."
pdflatex -shell-escape -interaction=nonstopmode main.tex >/dev/null

# 第三次编译 - 确保所有引用正确
echo "第三次 pdflatex 编译..."
pdflatex -shell-escape -interaction=nonstopmode main.tex >/dev/null

# 检查编译结果
if [ -f "main.pdf" ]; then
    echo "编译成功！生成了 main.pdf"
    # 清理临时文件（可选）
    echo "清理临时文件..."
    rm -f main.aux main.log main.blg main.out main.synctex.gz main.bbl main.toc main.lof main.lot
    echo "编译完成！"
else
    echo "编译失败，请检查错误信息"
    exit 1
fi
