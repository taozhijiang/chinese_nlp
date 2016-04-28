#一些汉语言处理的东西   

## segment 汉语言分词   
- 原理：HHM   
- 依赖：numpy scipy hhmlearn    
- 参考：   
  - [Itenyh版-用HMM做中文分词四：A Pure-HMM 分词器](http://www.52nlp.cn/itenyh%E7%89%88-%E7%94%A8hmm%E5%81%9A%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%9B%9B%EF%BC%9Aa-pure-hmm-%E5%88%86%E8%AF%8D%E5%99%A8)   
  - [自己写中文分词之（二）_用HMM模型实现分词](http://sbp810050504.blog.51cto.com/2799422/1251640)   
- TODO：   
  - 没有针对英文单词、标点等处理，也没有窗口化操作    

	
## 主题分类   
- 原理：LDA & Labled LDA   
- 依赖：gensim   
- 参考:
  - http://blog.itpub.net/16582684/viewspace-1253901/
  - https://radimrehurek.com/gensim/models/ldamodel.html
  - https://shuyo.wordpress.com/
- TODO:
  - 感觉效果不是很好啊

## LSI/LDA信息检索
- 原理：SVD奇异值分解
- 依赖：gensim
- TODO:
 - 原理比较简单，只有SVD，检索结果还是可以的，但是需要调整topic的参数，工程上以200-500为佳。

## 情感分析
- 原理: 基于统计的方式   
- 依赖：ntlk sklearn   
- 语料：某东的商品评论，好评-差评   
- 参考：   
  - [Python 文本挖掘：使用scikit-learn 机器学习包进行文本分类 ](http://rzcoding.blog.163.com/)

## 贝叶斯分类   
- 用C/C++重新实现后，发现内存占用率和运算速度比Python要块很多。   
- 通过Sogou的训练语料发现，10个分类下，10000特征词的分类准确率在75%左右，而在京东抓取的好评/差评语料训练后，测试分类精度达到91%左右。   

## 最大熵分类器   
- 原理：自行Google   
- 参考：   
  - http://www.fuqingchuan.com/2015/03/776.html   
  - http://www.nltk.org/_modules/nltk/classify/maxent.html   
  - http://www.umiacs.umd.edu/~hal/megam/index.html   
- NOTE：   
  - 基本是按照nltk的GIS算法翻译过来的，没有实现IIS，所以训练的速度非常的慢。
- MEGAM   
  - 添加了MEGAM部分，底层调用的megam是基于L-BFGS实现的，所以速度还是挺快的，有实用价值了。另外底层调用的megam是
二进制程序（32位）的，所以你的系统要支持32位的运行库（dnf install glibc.i686）

## 基于CRF的（NER命名实体识别）   
- 参考：[CRFSuite Manual](http://www.chokkan.org/software/crfsuite/manual.html)   
- TODO：   
   - 人家已经理论分析了CRF的效果会比贝叶斯和马尔科夫模型要好，而且CRF当前最主要的应用就是NLP的分词、序列标注和命名实体识别了。个人测试觉得，算法的收敛的速度很慢，所以模型只迭代训练了五百次。此外，现在的算法都十分的成熟了，而真正的壁垒在于数据，国内的开发比较的保守，公开的标注语料少之又少。人民日报的标注语料公开的部分不多，而且文字比较的书面和守旧，效果一般。   
   - 例子   
   > 还得从20年前中B-ORG 共I-ORG 召开十二大前夕说起。1982年6月27日至29日的中B-ORG 共I-ORG 十一届六中全会期间，印发了陈B-PER 云I-PER 撰写的《提拔培养中青年干部是当务之急》一文和他主持起草的《关于老干部离休退休问题座谈会纪要》。会后，部分与会人员留下来参加各省市自B-ORG 治I-ORG 区I-ORG 党I-ORG 委I-ORG 书记座谈会。7月2日，陈B-PER 云I-PER 在座谈会上讲话，强调干部队伍青黄不接的客观存在，不无担忧地说：提五十岁左右的人可能争论少些，提40岁左右的人，争论、怀疑会很多。提40岁以下的人，怀疑、争论会更多。既然如此，为什么“纪要”还是“特别写提四十岁以下的人这一句？”他自问自答：一是年富力强。二是有意识地培养。经过3年、5年、10年，有意识地培养，选出好的人。三是40岁以下的人中间有人才。四是只有40岁以下的人，才了解“文革”初期青年人当时的想法和表

## 基于同义词词林的消歧实现：   
- 原理：基于同义词词林的语料库反查，设定各个意项的评分。   
- 结果：不知道是这种方式的原因，还是评分函数优化的不合理，在标注的语料下，准确度大概44%左右。   

# 深度学习部分   
## 依赖和使用的深度学习库   
  - theano (CUDA optional)   
  - keras   
  - genism   


## 深度学习分词
- 参考：http://xccds1977.blogspot.com/2015/11/blog-post_25.html   
- 语料库：[北大和微软研究院的分词语料](http://www.sighan.org/bakeoff2005/)   
- TODO:   
  - 对于英文单词和数字的处理   
  - 加大神经网络的神经节点数目   
- 分词效果(LSTM 100-100 4类标注 15次迭代):   
> 中东 和平 的 建设者 、 中东 发展 的 推动 者 、 中东 工业化 的 助 推 者 、 中东 稳定 的 支持者 、 中东 民心 交融 的 合作 伙伴 —— 习近 平 主席 在 演讲 中 为 中国-中东关系 发展 指明 的 方向 ， 切合 地区 实际 情况 ， 照顾 地区 国家 关切 ， 为 摆 在 国际 社会 面前 的 “ 中东 之 问 ” 给 出 了 中国 的 答案 。 
> 2014年6 月 ， 习近 平在 中 阿 合作 论坛 北京 部长 级 会议 上 提出 ， 中 阿 共建 “ 一带 一路 ” ， 构建 以 能源 合作 为 主轴 ， 以 基础 设施 建设 、 贸易 和 投资 便利 化 为 两翼 ， 以 核能 、 航天 卫星 、 新 能源 三 大 高 新 领域 为 新 的 突破口 的 “ 1 + 2 + 3 ” 合作 格局 。 
> 在 此次 落马 的 16 人 里面 ， 级别 最高 的 是 连 城县委 原 书记 江国河 。 履历 显示 ， 江 国河 196 3年 出生 ， 龙岩市 永定县 高头乡 人 。 被 调查 时 ， 他 已 在 福建省 能源集团有限责任 公司 董事 、 纪委 书记 的 位子 上 干 了 两年 。 
> 机智堂 是 新 浪 手机 推出 的 新 栏目 ， 风趣 幽默 是 我们 的 基调 ， 直白 简单 地 普及 手机 技术 知识 是 我们 的 目的 。 我们 谈 手机 ， 也 谈 手 机 圈 的 有 趣事 ， 每月 定期 更新 ， 搞 机 爱好者 们 千万 不 能 错过 。 


## RNN-LSTM自动文本生成   
- 参考：   
  - [RNN Character Model - 2 Layer](https://github.com/ebenolson/pydata2015/blob/master/4%20-%20Recurrent%20Networks/RNN%20Character%20Model%20-%202%20Layer.ipynb)   
  - [char-rnn](https://github.com/karpathy/char-rnn)   
  - [lstm_text_generation](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)   
