# Image-Retrieval
Dataset: https://drive.google.com/file/d/1wVZjSByJxyhgIZGyqwabyUobuafQU5kP/view

In similarity, we implemented some difference-calculation method:

To call these method:

L1 - get_L1_score()

$$d_{L1}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{n} |u_i - v_i|$$

L2 - get_L2_score()

$$d_{L2}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$

Cosine Similarity - get_cosine_similarity()

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}}$$

Correlation coefficient - get_correlation_coefficient()

$$r = \frac{\sum_{i=1}^{n} (u_i - \bar{u})(v_i - \bar{v})}{\sqrt{\sum_{i=1}^{n} (u_i - \bar{u})^2 \sum_{i=1}^{n} (v_i - \bar{v})^2}}$$
