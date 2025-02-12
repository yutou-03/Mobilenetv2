每次修改更新流程：

1.如果远程仓库已有代码，建议先拉取最新内容，避免冲突：

git pull origin master

2.添加更改到暂存区

1. 在 VSCode 左侧的“源代码管理”选项卡中，查看更改的文件。

​    2.点击文件旁边的 `+` 号，将更改添加到暂存区。

​    或者使用终端命令：git add .

3.提交更改

1. 在“源代码管理”选项卡的输入框中填写提交信息。
2. 点击“提交”按钮。
   - 或者使用终端命令    git commit -m "你的提交信息"

4.推送更改到github

​      使用终端命令git push origin master