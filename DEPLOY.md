# 发布网页版

网页版文件位于仓库的 [`docs/`](docs/) 目录。将仓库设为公开后，在 GitHub 仓库页面依次打开 **Settings → Pages → Build and deployment**，将 **Source** 设为 **Deploy from a branch**，分支选 **main**、文件夹选 **/docs**，然后点 **Save**。

部署成功后，预计网址为 <https://xinyang-guo.github.io/guzheng-tuning-drift-model/>；最终以 GitHub Pages 设置页显示的网址为准。打开网页后可分别选择岭回归和随机森林，检查 21 根弦的预测图表和手机布局。

**公开前请核对数据。** 当前私有仓库根目录已有 `古筝调音数据.xlsx`。把此仓库设为公开，会让访客看到当前文件以及 Git 历史中的工作簿。只删除工作簿或把它加入 `.gitignore`，不会从 Git 历史中移除它。仓库可见性由仓库所有者自行更改。

操作参考：[GitHub 官方说明：配置 Pages 发布来源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)。
