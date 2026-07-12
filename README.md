# Image-Retrieval
Dataset: https://drive.google.com/file/d/1wVZjSByJxyhgIZGyqwabyUobuafQU5kP/view

In similarity, we implemented some difference-calculation method:
To call these method:

<br>L1 - get_L1_score()
<br>$$d_{L1}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{n} |u_i - v_i|$$

<br>L2 - get_L2_score()
<br>$$d_{L2}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$

<br>Cosine Similarity - get_cosine_similarity()
<br>$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}}$$

<br>Correlation coefficient - get_correlation_coefficient()
<br>$$r = \frac{\sum_{i=1}^{n} (u_i - \bar{u})(v_i - \bar{v})}{\sqrt{\sum_{i=1}^{n} (u_i - \bar{u})^2 \sum_{i=1}^{n} (v_i - \bar{v})^2}}$$
