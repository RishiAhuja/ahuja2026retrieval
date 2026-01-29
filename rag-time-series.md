Abstract
Time series forecasting uses historical data to
predict future trends, leveraging the relation-
ships between past observations and available
features. In this paper, we propose RAFT, a
retrieval-augmented time series forecasting
method to provide sufficient inductive biases and
complement the model’s learning capacity. When
forecasting the subsequent time frames, we di-
rectly retrieve historical data candidates from the
training dataset with patterns most similar to the
input, and utilize the future values of these can-
didates alongside the inputs to obtain predictions.
This simple approach augments the model’s ca-
pacity by externally providing information about
past patterns via retrieval modules. Our empirical
evaluations on ten benchmark datasets show that
RAFT consistently outperforms contemporary
baselines with an average win ratio of 86%.
Use retrieved results alongside the input
Forecast
Retrieve relevant historical patterns
Figure 1. Illustration of a motivating example of retrieval in time-
series forecasting.
1. Introduction
Accurately predicting future trends is crucial for making
informed decisions in various fields. Time series forecast-
ing, which analyzes past data to anticipate future outcomes,
plays a vital role in diverse areas like climate model-
ing (Zhu & Shasha, 2002), energy (Mart´ ın et al., 2010),
economics (Granger & Newbold, 2014), traffic flow (Chen
et al., 2001), and user behavior (Benevenuto et al., 2009).
By providing reliable predictions, it empowers us to develop
effective strategies and policies across these domains.
Over the past decade, deep learning models such as
CNNs (Bai et al., 2018; Borovykh et al., 2017) and
RNNs (Hewamalage et al., 2021) have proven their
*Equal contribution 1Department of AI Convergence, GIST,
Gwangju, South Korea. This work is done while the author was
in KAIST. 2School of Computing, KAIST, Daejeon, South Ko-
rea 3Data Science for Humanity Group, Max Planck Institute for
Security and Privacy, Bochum, Germany 4Google Cloud AI, Sun-
nyvale, United States. Correspondence to: Jinsung Yoon <jin-
sungyoon@google.com>.
Proceedings of the 42 nd International Conference on Machine
Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025
by the author(s).
effectiveness in capturing patterns of change in historical
observations, leading to the development of various deep
learning models tailored for time series forecasting. Espe-
cially, the advent of attention-based transformers (Vaswani
et al., 2017) has made a significant impact on the time
series domain. The architecture has shown to be effective in
modeling dependencies between inputs, resulting in variants
like Informer (Zhou et al., 2021), AutoFormer (Wu et al.,
2021), and FedFormer (Zhou et al., 2022). Additionally,
recent methods utilize time series decomposition (Wang
et al., 2023), which isolates trends or seasonal patterns,
and multi-periodicity analysis which involves downsam-
pling/upsampling of the series at various periods (Lin et al.,
2024; Wang et al., 2024). Furthermore, lightweight models
like multi-layer perceptrons (MLP) have demonstrated
strong performance along with these decomposition
techniques and multi-periodicity analysis (Chen et al., 2023;
Zeng et al., 2023; Zhang et al., 2022).
However, real-world time series exhibit complex, non-
stationary patterns with varying periods and shapes. These
patterns may lack inherent temporal correlation and arise
from non-deterministic processes, resulting in infrequent
repetitions and diverse distributions (Kim et al., 2021). This
raises concerns about the effectiveness of models in extrap-
olating from such infrequent patterns. Moreover, the advan-
tages of indiscriminately memorizing all patterns, including
noisy and uncorrelated ones, are questionable in terms of
both generalizability and efficiency (Weigend et al., 1995).
We show an advancement in time-series forecasting mod-
els by expanding the models’ capacity (implicitly via the
trained weights) to learn patterns. We directly provide infor-
mation about historical patterns that are complex to learn,
as a way of bringing relevant information via the input
to reduce the burden on the forecasting model. Inspire
by the retrieval-augmented generation (RAG) approaches
used in large language models (Lewis et al., 2020), our
method retrieves similar historical patterns from the training
dataset based on given inputs and utilizes them along with
the model’s learned knowledge to forecast the next time
frame (see Figure 1).
Our new approach, Retrieval-Augmented Forecasting of
Time-series (RAFT), offers two key advantages. First, by
directly utilizing retrieved information, the useful patterns
from the past become explicitly available at inference time,
rather than utilizing them via the learned information in
model weights. Learning hence covers patterns that lack
temporal correlation or do not share common characteristics
with other patterns, thereby reducing the learning burden
and enhancing generalizability. Second, even if a pattern
rarely appears in historical data and is difficult for the model
to memorize, the retrieval module allows the model to easily
leverage historical patterns when they reappear (Miller
et al., 2024; Laptev et al., 2017).
We demonstrate that the proposed judiciously-designed
inductive bias, implemented through a simple retrieval
module, enables an MLP architecture to achieve strong
forecasting performance. Inspired by existing literature that
downsamples series at various period intervals (Lin et al.,
2024; Wang et al., 2024), RAFT also generates multiple
series by downsampling the given series at different periods
and attaches a retrieval module to each series. This allows
for effectively capturing both short-term and long-term pat-
terns for more accurate forecasting. As demonstrated on ten
time-series benchmark datasets, RAFT outperforms other
contemporary baselines with an average win ratio of 86%.
Overall, our contributions can be summarized as follows:1
• We propose a retrieval-augmented time series forecast-
ing method, RAFT, which retrieves observations with
similar temporal patterns from the training dataset
and effectively leverage retrieved patterns for future
predictions.
• Our empirical studies on ten different benchmark
datasets show that RAFT outperforms other contem-
porary baselines with an average win ratio of 86%.
• We further explore the scenarios where retrieval mod-
ules can be beneficial for forecasting by conducting
analyses using synthetic and real-world datasets.
2. Related Work
2.1. Deep learning for time-series forecasting
A large body of research employs deep learning for
time-series forecasting. Existing methods can be broadly
categorized based on the employed architecture. Prior to the
advent of transformers (Vaswani et al., 2017), time series
analysis often relied on CNNs to capture local temporal pat-
terns (Bai et al., 2018; Borovykh et al., 2017) or RNNs to
model sequential dependencies (Hewamalage et al., 2021).
Following the advent of transformers, several approaches
emerged to better tailor the transformer architecture for
time-series forecasting. For example, LogTrans (Li et al.,
2019) used a convolutional self-attention layer, while In-
former (Zhou et al., 2021) employed a ProbSparse attention
module along with a distilling technique to efficiently
reduce network size. Both Autoformer (Wu et al., 2021) and
FedFormer (Zhou et al., 2022) decomposed time series into
components like trend and seasonal patterns for prediction.
Despite advancements in transformer-based models, (Zeng
et al., 2023) reported that even a simple linear model
can achieve strong forecasting performance. Sub-
sequently, lightweight MLP-based time-series models
such as TiDE (?), TSMixer (Chen et al., 2023), and
TimeMixer (Wang et al., 2024) were introduced with the ad-
vantages in both forecasting latency and training efficiency.
These models utilize various approaches such as series de-
composition similar to transformer-based studies (Zeng
et al., 2023) or introduced multi-periodicity analysis by
downsampling or upsampling the series at various period
intervals (Lin et al., 2024), to accurately extract the rele-
vant information from time-series for MLPs to effectively
fit on them. Recently, several studies have constructed a
large time-series databases to build large foundation models,
achieving strong zero-shot and few-shot performance (Das
et al., 2024; Woo et al., 2024).
Our proposed RAFT is based on a shallow MLP architecture,
following simplicity and efficiency motivations. Through
the retrieval module, the model retrieves subsequent pat-
terns that follow the patterns most similar to the current
input from the single time series, allowing it to reference
past patterns for future predictions without the burden of
memorizing all temporal patterns during training. Our re-
trieval differs from transformer variants that typically learn
relationships only within a fixed lookback window. RAFT
goes beyond the lookback window by retrieving relevant
data points from the entire time series and incorporating
them into the input.
2.2. Retrieval augmented models
Retrieval-augmented models typically work by first retriev-
ing relevant instances from a dataset based on a given input.
Then, they combine the input with these retrieved instances
to generate a prediction. Retrieval-augmented generation
(RAG) in natural language domain is an active research
area that utilizes this scheme. (Lewis et al., 2020; Guu
et al., 2020). RAG retrieves document chunks from external
corpora that are relevant to the input task, helping large lan-
guage models (LLMs) generate responses related to the task
without hallucination (Shuster et al., 2021; Borgeaud et al.,
2022). This not only supplements the LLM’s limited prior
knowledge but also enables the LLM to handle complex,
knowledge-intensive tasks more effectively by providing
additional information from the retrieved documents (Gao
et al., 2023).
Beyond natural language processing, retrieval-augmented
models have also been used to solve structured data prob-
lems. A simple illustrative example is the K-nearest neigh-
bor model (Zhang, 2016). Other approaches have intro-
duced kernel-based neighbor methods (Nader et al., 2022),
prototype-based approaches (Arik & Pfister, 2020), or con-
sidered all training samples as retrieved instances (Kossen
et al., 2021). More recently, models leveraging attention-
like mechanisms have incorporated the similarity between
retrieved instances and the input into the prediction, achiev-
ing superior performance compared to traditional deep tab-
ular models (Gorishniy et al., 2024). There also exists a
method that has explored the potential of retrieving similar
entities in time-series forecasting, involving multiple time
series entities (Iwata & Kumagai, 2020; Yang et al., 2022).
Assuming the training set contains various types of time
series entities, they aggregate the information needed for
each entity’s prediction based on the similarities across all
time series entities.
In this paper, we aim to demonstrate that retrieval can be ef-
fective, even when applied to the single time-series. Similar
to how RAG supplements LLMs with additional informa-
tion for knowledge-intensive tasks, our approach seeks to
reduce the learning complexity in time-series forecasting.
Instead of forcing the model to learn every possible com-
plex pattern, the retrieval module provides information that
simplifies the learning process.
3. Method
3.1. Overview
Problem formulation. Given a single time series
S ∈ RC×T of length T with C observed variates (i.e.,
channels), RAFT utilizes historical observation x ∈RC×L
to predict future values y ∈RC×F that is close to the actual
future values y0 ∈RC×F
. Ldenotes look-back window
size and F denotes forecasting window size.
Given an input x, RAFT utilizes a retrieval module to find
the most relevant patch from S. Then, the subsequent
patches of the relevant patch are retrieved as additional
information for forecasting. The retrieval process follows
an attention-like structure, where the importance weights are
calculated based on the similarity between the input and the
patches, and the retrieved patches are aggregated through a
weighted sum (Sec. 3.2). The main difference of our model
from attention-based forecasting models, such as transform-
ers, lies in its ability to retrieve relevant data from the entire
time series rather than relying on a fixed lookback window.
Since the time series shows distinct characteristics across
periods, we utilize the retrieval modules into multiple peri-
ods. RAFT generates multiple time series by downsampling
the time series S with different periods and applies the re-
trieval module to each time series. The retrieval results from
multiple series are processed through linear projection and
aggregated by summation. Finally, the input and the aggre-
gated retrieval result are concatenated and passed through
a linear model to produce the final prediction (Sec. 3.3).
Details of each component are described below.
3.2. Retrieval module architecture
We transform the time series S to be appropriate for re-
trieval. First, we find all key patches within S that are to be
compared with given x ∈RC×L. Using the sliding window
method of stride 12, we extract patches of window size L
and define this collection as K= {k1,...,kT−(L+F)+1},
where i indicates the starting time step of the patch
ki ∈RC×L. Note that any patch that overlaps with the
given xmust be excluded from Kduring the training phase.
Then, we find all value patches that sequentially follow each
key patch ki ∈Kin the time series. We define the collection
of value patches as V∈{v1,...,vT−(L+F)+1}, where each
vi ∈RC×F sequentially follows after ki in the time series.
After preparing the key patch set Kand value patch set V
for retrieval, we use the input x as a query to retrieve similar
key patches along with their corresponding value patches
with following steps. We first account for the distributional
deviation between the query, key, and value patches used
in the retrieval process. Let us define x = {xt}t∈{1,...,L},
where xt ∈RC denotes the values of Cvariates at t-th time
step within the input x (i.e., xt = {xt
1,...,xt
C}). Inspired
by existing literature (Zeng et al., 2023), we treat the final
time step value in each patch as an offset and subtract this
value from the patch as a form of preprocessing to make the
patterns more meaningful to compare:
ˆ
x = {xt
−xL}t∈{1,...,L}, (1)
whereˆ
x represent the input queries with the offset sub-
tracted. Similarly, we subtract the offset from all key patches
ki ∈Kand vi ∈V, denoting them asˆ
ki ∈ˆ
Kandˆ
vi ∈ˆ
V,
respectively. Then, we calculate the similarity ρi between
givenˆ
x and all key patches inˆ
Kusing similarity function s:
ˆ
ˆ
ρi = s(ˆ
x,
ki),
ki ∈ˆ
K. (2)
Here, we use Pearson’s correlation as a similarity function
sto exclude the effects of scale variations and value offsets
2The stride can be adjusted according to the demand of compu-

n the time series, focusing on capturing the increasing and
decreasing tendencies3. We then retrieve the patches with
top-mcorrelation values:
where linear projections f maps RL to RF
, gmaps RF to
RF
, hmaps R2F to RF, and ⊕represents concatenation
operation.
J= arg top-m({ρi |1 ≤i≤|ˆ
K|}), (3)
where J denotes the indices of top-m patches. Given
temperature τ, we calculate the weight of value patches
with following equation:
wi =
exp (ρi /τ)
j∈J exp (ρj /τ) , if i∈J
0. otherwise
(4)
Note that this is equivalent to conduct SoftMax only with
top-m correlation values. Finally, we obtain the final
retrieval result˜
v ∈RC×F as the weighted sum of value
patches:
˜
v =
i∈{1,...,|ˆ
V|}
wi·
ˆ
vi. (5)
Figure 2 illustrates the architecture of our retrieval module.
3.3. Forecast with retrieval module
Single period. Consider the given input x ∈RC×L and the
retrieved patch˜
v ∈RC×F. Similar to the retrieval module,
we subtract the offset from x and defineˆ
x as the input with
the offset removed. Next, we concatenate f(ˆ
x) with g(˜
v),
and process concatenated result through hto obtainˆ
y:
ˆ
y= h(f(ˆ
x) ⊕g(˜
v)), (6)
3See Appendix C.1 for comparison results with different simi-
larity metrics.
Multiple periods. Time series at different periods display
unique characteristics – patterns in a small time window
typically reveal local patterns, while patterns in a large time
window might correspond to global trends. We extend the
retrieval process to consider nperiods P. For each p∈P,
we downsample the query x, all key patches in K, and all
value patches in Vof period 1 by average pooling with
period p. This results in x(p) ∈RC×⌊L
p ⌋
, K(p), and V(p)
as the respective query, key patch set, and value patch set
for period p, where a key patch k(p)
i ∈ RC×⌊L
p ⌋ and a
value patch v(p)
i ∈RC×⌊F
p ⌋. Then, we conduct the retrieval
process described in Sec. 3.2 using x(p)
, K(p), and V(p)
,
and obtain the retrieval result˜
v(p) ∈ RC×⌊F
p ⌋ for each
p. Each˜
v(p) is processed through a linear layer g(p) to
project all retrieval results in the same embedding space,
mapping R⌊F
p ⌋to RF, respectively. Finally, we concatenate
ˆ
x with sum of linear projections and process it through linear
predictor h, which replaces Eq. 6 to following equation:
ˆ
y= h(f(ˆ
x) ⊕
p∈P
g(p)(˜
v(p))) (7)
Denotingˆ
yt as the value at the t-th time step withinˆ
y, we
restore the original offset by adding xL toˆ
y, resulting in
the final forecast y:
y= {ˆ
yt + xL}t∈{1,...,F}. (8)
We train the model by minimizing the following MSE loss:
L= MSE(y, y0) (9)
Figure 3 illustrates our model’s forecasting process with
multiple periods of retrieval. Hyperparameters such as m
are chosen based on the performance in the validation set.
4. Experiments
We evaluate RAFT across multiple time series forecasting
benchmark datasets. We analyze how our proposed retrieval
module contributes to performance improvement in time-
series forecasting, and in which scenarios retrieval is partic-
ularly beneficial. The full results, visualizations, and addi-
tional analyses of our model are provided in the Appendix.
4.1. Experimental settings
Datasets. We consider ten different benchmark datasets,
each with a diverse range of variates, dataset lengths, and
frequencies: (1-4) The ETT dataset contains 2 years of
electricity transformer temperature data, divided into four
subsets—ETTh1, ETTh2, ETTm1, and ETTm2 (Zhou et al.,
2021); (5) The Electricity dataset records household electric
power consumption over approximately 4 years (Trindade,
2015); (6) The Exchange dataset includes the daily exchange
rates of eight countries over 27 years (1990–2016) (Lai et al.,
2018); (7) The Illness dataset includes the weekly ratio of pa-
tients with influenza-like illness over 20 years (2002-2021)4;
(8) The Solar dataset contains 10-minute solar power fore-
casts collected from power plants in 2006 (Liu et al., 2022a);
(9) The Traffic dataset contains hourly road occupancy rates
on freeways over 48 months5; and (10) The Weather dataset
4https://gis.cdc.gov/grasp/fluview/
fluportaldashboard.html
5https://pems.dot.ca.gov/
consists of 21 weather-related indicators in Germany over
one year6. Data summary is provided in the Appendix A.
Baselines. We compare against 9 contemporary time-series
forecasting baselines, including: (1) Autoformer (Wu
et al., 2021), (2) Informer (Zhou et al., 2021), (3) Sta-
tionary (Liu et al., 2022b), (4) Fedformer (Zhou et al.,
2022), and (5) PatchTST (Nie et al., 2023), all of which
use Transformer-based architectures; (6) DLinear (Zeng
et al., 2023), which are lightweight models with simple
linear architectures; (7) MICN (Wang et al., 2023), which
leverages both local features and global correlations
through a convolutional structure; (8) TimesNet (Wu
et al., 2023), which utilizes Fourier Transformation to
decompose time-series data within a modular architecture;
and (9) TimeMixer (Wang et al., 2024), which utilizes
decomposition and multi-periodicity for forecasting7
.
Implementation details. RAFT employs the retrieval mod-
ule with following detailed settings. The periods are set to
{1,2,4}(n= 3), following existing literature (Wang et al.,
2024), and the temperature τis set to 0.1. Batch size is set to
32. The initial learning rate, the number of patches used in
the retrieval (m), and the size of the look-back window (L)
are determined via grid search based on performance on the
validation set, following the prior work (Wang et al., 2024).
For fair comparison, hyper-parameter tuning was performed
for both our model and all baselines using the validation
set. The learning rate is chosen from 1e-5 to 0.05, look back
6https://www.bgc-jena.mpg.de/wetter/
7We compare our model with general time-series forecasting
models. Other retrieval-based time-series models mentioned in the
related work assume the presence of multiple time-series instances,
which are outside the scope of our study.
window size from {96,192,336,720}, and the number of
patches used in retrieval mfrom {1,5,10,20}. The cho-
sen values of each setting are presented in the Appendix B.
For implementation, we referred to the publicly available
time-series repository (TSLib)8. For all experiments, the
average results from three runs are reported, with each exper-
iment conducted on a single NVIDIA A100 40GB GPU. For
more details about the computational complexity analysis
of RAFT, see Appendix D.
Evaluation. We consider two metrics for evaluation: MSE
and MAE. We varied the forecasting horizon length to mea-
sure performance (i.e., F= 96, 192, 336, 720), and each ex-
periment setting was run with three different random seeds
to compute the average results. For the Illness dataset, fore-
casting horizons of 24, 36, 48, and 60 are used, following
the prior work (Nie et al., 2023; Wang et al., 2024). The
evaluation was conducted in multivariate settings, where
both the input and forecasting target have multiple channels.
4.2. Experimental results on forecasting benchmarks
Table 1 presents comparisons between the performance
of time series forecasting methods and RAFT. The results
represent the average MSE performance evaluated across
different forecasting horizon lengths. We observe that
our model consistently outperforms other contemporary
baselines on average, supporting the effectiveness of
retrieval in time series forecasting. Full results and
comparisons using a different evaluation metric (i.e., MAE)
are provided in Appendix G.
5. Discussions
In this section, we explore scenarios where retrieval shows
substantial advantage by empirically analyzing its effect,
8https://github.com/thuml/
Time-Series-Library
using both benchmark time series datasets and synthetic
time series datasets.
5.1. Better retrieval results lead to better performance.
Two criteria are important for our retrieval method to
enhance the forecasting performance. First, the value
patches Videntified through the similarity between the input
query x and key patches Kshould closely match the actual
future value y0 which sequentially follows the input query.
Second, the model should efficiently leverage the informa-
tion in the value patches for forecasting. From these, we can
draw the insight that higher similarity between input query
and key patches (i.e., key similarity) will lead to the higher
similarity between the actual value and value patches (i.e.,
value similarity), eventually resulting in better performance.
Figure 4 presents the correlation analysis conducted on the
ETTh1 dataset. Figure 4a shows that retrieving key patches
with higher similarity leads to value patches that are more
closely aligned with the actual future value. Figure 4b il-
lustrates that the value patches with greater similarity to
the actual future values tend to improve RAFT’s perfor-
mance more significantly. This trend is also consistent
across datasets; datasets with higher key similarity show
higher value similarity, resulting in larger performance gains.
Spearman’s correlation coefficient validate this trend, show-
ing a correlation of 0.60 between key similarity and value
similarity, and a correlation of−0.54 between value simi-
larity and performance gain across datasets. The negative
correlation with performance is due to the use of MSE as
the metric (lower the better). These results demonstrate
that better retrieval results from the retrieval module lead to
improved performance of RAFT.
5.2. Retrieval is helpful when rare patterns repeat.
RAFT can complement scenarios where a particular pattern
does not frequently appear in the training dataset, making it
6
Retrieval Augmented Time Series Forecasting
(a) Scatter plot of key and value similarity (b) Scatter plot of value similarity and MSE change (%)
Figure 4. Analysis of the correlation between (a) the key similarity and value similarity, and (b) the value similarity and model performance
changes measured by MSE (%). Each dot represents each input patch from the ETTh1 test dataset. Key similarity refers to the average
similarity between input query (x) and all retrieved key patches (K). Value similarity refers to the average similarity between actual future
value (y0) and all retrieved value patches (V).
difficult for the model to memorize. By utilizing retrieved
information, the model can overcome this challenge. To an-
alyze this effect, we conducted experiments using synthetic
time series datasets.
Synthetic data generation with autoregressive model.
The synthetic time series was constructed by combining
three components: trend, seasonality, and event-based short-
term patterns. Trend and seasonality were generated using
sinusoidal functions with varying periods, amplitudes, and
offsets, representing long-term consistent patterns. Short-
term patterns, modeled as event-based dynamics, were cre-
ated using an autoregressive model:
xt =
20
ϕixt−i + ϵt, (10)
i=1
where ϕi are autoregressive parameters, and ϵt is noise sam-
pled from a uniform distribution. The short-term pattern
length was fixed at 200. To test retrieval effectiveness for
rare patterns, we generated three distinct short-term patterns
and varied their frequency in the training dataset. Forecast-
ing accuracy (MSE) was evaluated when each short-term
pattern appeared in the test set, with input and forecasting
horizon lengths fixed at 96. Additional dataset details and
figures are available in Figure 5a and Appendix E.
Results. Table 2 presents the number of occurrences of
the short-term patterns and the corresponding performance
of RAFT with and without retrieval, as well as baseline
models. Note that, in this experiment, we did not consider
multiple periods in order to isolate the effect of retrieval,
so RAFT without retrieval has an identical structure to
the NLinear (Zeng et al., 2023). The results show that our
model, utilizing retrieval, consistently outperformed the
model without retrieval on the synthetic dataset; 9.2∼14.7%
increase in performance depending on the pattern occur-
rences. Notably, as the pattern occurrences decreased, the
Table 2. Analysis between forecasting accuracy and the rarity of
the pattern over the synthetic time series with an autoregressive
model. Forecasting accuracy was evaluated using MSE, averaged
across 120 different time series and short-term patterns. The num-
bers in the last row indicate the ratio by which the MSE decreases
when retrieval is appended.
Pattern occurrences 1 2 4
TimeMixer 0.2360 0.2166 0.2276
TimesNet 0.2282 0.1970 0.1925
MICN 0.2285 0.2331 0.2033
DLinear 0.2640 0.2552 0.2502
RAFT without Retrieval 0.2590 0.2310 0.2344
RAFT with Retrieval 0.2209 0.2064 0.2128
MSE decrease ratio -14.7% -10.7% -9.2%
reduction in MSE was more significant. Similar to RAFT
without retrieval, the baseline models exhibited a decrease
in performance as the pattern occurrences decreased. When
we also visualize the predictions of models with and without
retrieval modules over the rare pattern (see Figure 5b),
the model utilizing retrieval aligns well with the pattern’s
periodicity and offset during forecasting, while the model
relying solely on learning fails to capture these aspects. This
suggests that the model struggles to learn rare patterns, and
the retrieval module effectively complements this deficiency.
5.3. Retrieval is helpful when patterns are temporally
less correlated.
If short-term patterns are very similar across time, there’s
less unique information for the model to learn, making it
easier to achieve accurate predictions. On the other hand,
if the short-term patterns in time series data are similar to
a random walk without any specific temporal correlation,
the model would need to memorize all changes within

short-term pattern for accurate forecasting. Based on this
hypothesis, we expect the retrieval module to be especially
helpful when patterns are temporally less correlated, as
retrieval can easily detect similarities between patterns that
temporal correlation alone cannot capture. We again use
the synthetic dataset for validation.
Synthetic data generation with random walk model.
Instead of generating short-term patterns using the autore-
gressive model as before, we utilize random walk-based
change patterns, following the equation:
xt = xt−1 + ϵt. (11)
The step size for the walk ϵt was sampled from a uniform
distribution within the range of [-20, 20]. The generated
short-term patterns were then inserted into the training data,
as in the previous synthetic time-series approach.
Table 3. Forecasting accuracy over the rarity of the pattern. Syn-
thetic time series with random walk based patterns (temporally
less correlated) is used. Forecasting accuracy was evaluated using
MSE, averaged across 120 different time series and short-term
patterns. The numbers in the last row indicate the ratio by which
the MSE decreases when retrieval is appended.
series data. Again, the retrieval module improves perfor-
mance across all cases, particularly for rare patterns. Fur-
thermore, the performance improvement is more signifi-
cant for temporally less correlated patterns (16.0∼31.5%
decrease of MSE depending on pattern occurrences), com-
pared to temporally more correlated ones shown in Table 2
(9.2∼14.7%). The baseline models exhibited a similar trend
to that observed in Table 2, while the performance gap
compared to RAFT with retrieval has become more signif-
icant. This confirms that the proposed retrieval module is
more beneficial when dealing with temporally less corre-
lated or near-random patterns that are more challenging for
the model to learn.
5.4. Retrieval is also helpful for Transformer-variants
We investigate the effectiveness of the retrieval module
Transformer-variants, using AutoFormer. Instead of modi-
fying the internal Transformer architecture to integrate our
retrieval module, we directly added retrieval results to Aut-
oFormer’s predictions at the final stage. Table 4 demon-
strates that our retrieval module successfully enhances the
forecasting performance of the Transformer-based model,
highlighting its broader applicability to other architectures.
Pattern occurrences 1 2 4
TimeMixer 0.2863 0.2305 0.2249
TimeNet 0.2448 0.1877 0.1938
MICN 0.2536 0.2445 0.2450
DLinear 0.3175 0.2059 0.2798
RAFT without retrieval 0.2694 0.2649 0.1894
RAFT with retrieval 0.1845 0.1818 0.1592
MSE decrease ratio -31.5% -31.4% -16.0%
Table 4. Performance comparison between AutoFormer and Aut-
oFormer with our proposed retrieval module. The average MSE
across different forecasting horizon lengths is reported.
ETTh1 ETTh2 ETTm1 ETTm2
Autoformer 0.496 0.450 0.588 0.327
+ Retrieval 0.471 0.444 0.454 0.326
Results. Table 3 shows the results of applying the same
experiment as in Table 2, but with different synthetic time-
8
Retrieval Augmented Time Series Forecasting
6. Conclusion
In this paper, we introduce RAFT, a time-series forecasting
method that leverages retrieval from training data to aug-
ment the input. Our retrieval module lessens the model to
absorb all unique patterns in its weights, particularly those
that lack temporal correlation or do not share common char-
acteristics with other patterns. This overall is demonstrated
as an effective inductive bias for deep learning architectures
for time-series. Our extensive evaluations on numerous real-
world and synthetic datasets confirm that RAFT achieves
performance improvements over contemporary baselines.
As various retrieval-based models are being proposed, there
remains room for improvement in retrieval techniques
specifically tailored for time-series data (beyond the simple
approaches used), including determining when, where, and
how to apply retrieval based on dataset characteristics and
capture more complex similarity measures that depend on
nonlinear and nonstationary characteristics. Our work is
expected to open new avenues in the time-series forecasting
field through the use of retrieval-augmented approaches.
Acknowledgement
We would like to thank Tomas Pfister and Dhruv Madeka
for their valuable feedback during the review of our paper.
Han, Lee, and Cha were partially supported by the National
Research Foundation of Korea grant (RS-2022-00165347).
Han was partially supported by GIST (AI-based Research
Scientist Project).
Impact Statement
This paper advances time series forecasting by enhancing
deep learning models with the retrieval of information from
historical training data. By reducing the learning burden
and mitigating memorization, this approach improves the
forecasting performance in various real-world applications,
such as weather prediction and financial analysis. We be-
lieve this work offers a new perspective on integrating the
concept of retrieval into the time-series domain.
References
Arik, S. O. and Pfister, T. Protoattend: Attention-based
prototypical learning. Journal of Machine Learning Re-
search, 21(210):1–35, 2020.
Bai, S., Kolter, J. Z., and Koltun, V. An empirical evalua-
tion of generic convolutional and recurrent networks for
sequence modeling. arXiv preprint arXiv:1803.01271,
2018.
Benevenuto, F., Rodrigues, T., Cha, M., and Almeida, V.
Characterizing user behavior in online social networks.
In Proceedings of the 9th ACM SIGCOMM Conference
on Internet Measurement, pp. 49–62, 2009.
Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford,
E., Millican, K., Van Den Driessche, G. B., Lespiau, J.-B.,
Damoc, B., Clark, A., et al. Improving language models
by retrieving from trillions of tokens. In International
conference on machine learning, pp. 2206–2240. PMLR,
2022.
Borovykh, A., Bohte, S., and Oosterlee, C. W. Condi-
tional time series forecasting with convolutional neural
networks. stat, 1050:16, 2017.
Chen, C., Petty, K., Skabardonis, A., Varaiya, P., and Jia, Z.
Freeway performance measurement system: mining loop
detector data. Transportation research record, 1748(1):
96–102, 2001.
Chen, S.-A., Li, C.-L., Arik, S. O., Yoder, N. C., and Pfister,
T. Tsmixer: An all-mlp architecture for time series fore-
casting. Transactions on Machine Learning Research,
2023.
Das, A., Kong, W., Sen, R., and Zhou, Y. A decoder-
only foundation model for time-series forecasting. In
Forty-first International Conference on Machine Learn-
ing, 2024.
Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y.,
Sun, J., and Wang, H. Retrieval-augmented generation
for large language models: A survey. arXiv preprint
arXiv:2312.10997, 2023.
Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D.,
Kotelnikov, A., and Babenko, A. Tabr: Tabular deep
learning meets nearest neighbors. In The Twelfth Interna-
tional Conference on Learning Representations, 2024.
Granger, C. W. J. and Newbold, P. Forecasting economic
time series. Academic press, 2014.
Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.
Retrieval augmented language model pre-training. In
International conference on machine learning, pp. 3929–
3938. PMLR, 2020.`
Hewamalage, H., Bergmeir, C., and Bandara, K. Recur-
rent neural networks for time series forecasting: Current
status and future directions. International Journal of
Forecasting, 37(1):388–427, 2021.
Iwata, T. and Kumagai, A. Few-shot learning for time-series
forecasting. arXiv preprint arXiv:2009.14379, 2020.
Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., and Choo, J.
Reversible instance normalization for accurate time-series
forecasting against distribution shift. In International
Conference on Learning Representations, 

