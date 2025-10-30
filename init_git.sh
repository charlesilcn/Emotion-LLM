#!/bin/bash

# 初始化Git仓库
git init

# 配置Git用户信息
echo "请输入您的GitHub用户名:"
read username
git config --global user.name "$username"

echo "请输入您的GitHub邮箱:"
read email
git config --global user.email "$email"

# 添加所有文件到暂存区
git add .

# 创建初始提交
git commit -m "Initial commit"

echo "Git仓库初始化成功！"
echo "接下来，您需要在GitHub上创建一个新仓库，然后运行以下命令："
echo "git remote add origin https://github.com/您的用户名/仓库名.git"
echo "git push -u origin main"