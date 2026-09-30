# 怎么把网页放到 GitHub 上

网页已经在仓库的 [`docs/`](docs/) 里，不用再上传一遍，也不用让自己的电脑一直开着。现在在线地址还没有启用。

**先确认一件事：**这个仓库根目录里有 `古筝调音数据.xlsx`。如果直接把仓库公开，别人也能看到这份文件和 Git 历史里的版本。只删除文件或加入 `.gitignore`，不能抹掉历史。如果原始数据不适合公开，应先用不含这些数据和历史记录的新仓库发布网页。

决定公开当前仓库后，按下面做：

1. 打开仓库的 **Settings → Pages**。
2. 在 **Build and deployment** 里，把 **Source** 选为 **Deploy from a branch**。
3. 分支选 **main**，文件夹选 **/docs**，点 **Save**。
4. 等 Pages 显示部署成功，再打开 <https://xinyang-guo.github.io/guzheng-tuning-drift-model/>。如果仍是 404，可以查看仓库的 **Actions** 中有没有部署失败记录。

打开后试一次：填入温湿度，分别选岭回归和随机森林，确认 21 根弦都能显示，并试试下载 CSV。

GitHub 的操作说明：[配置 Pages 发布来源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。
