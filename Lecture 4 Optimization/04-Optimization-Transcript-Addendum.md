# Lecture 4 Transcript Addendum

## When Flexibility Becomes Fragility: The 100-Sales/44-Effects Experiment

Let us make the idea of a constraint concrete. Imagine that we run a small appraisal office. We have only 100 recent home sales, but we want our model to be sophisticated, so we take the eight original measurements and add every pairwise interaction and squared term. We now have 44 possible effects to estimate from only 100 observations. Nothing in ordinary least squares prevents us from doing this. In fact, the unconstrained optimizer is delighted to use all of that flexibility if it can reduce training error.

Before we run the experiment, what do you expect to happen? More features make the model more expressive, so the training fit can only improve. But does that guarantee better predictions for new districts? This is the tension we want to see.

*Run the constraint experiment cell.*

The first two panels show predictions for the same held-out districts. The horizontal axis is the value recorded in the dataset, and the vertical axis is the model's prediction. The dashed diagonal is where a perfect prediction would lie. In the unconstrained panel, the cloud is widely dispersed and several predictions fall outside the range ever recorded in this dataset. The model has found a collection of large coefficients that fits peculiarities of its 100 training observations, but those coefficients do not travel well to new data.

Now look at the norm-constrained model. We solve the same prediction problem, but we limit the total magnitude of the coefficient vector. Geometrically, we are telling the optimizer that it may search only inside a ball: $\lVert w\rVert_2\le R$. The constraint does not tell the model what the correct coefficients are. It simply rules out solutions that depend on an extreme collection of positive and negative effects cancelling one another.

For this representative sample, the unconstrained model has a test MSE of about 1.04 and a coefficient norm of about 5.89. The constrained model reduces the test MSE to about 0.68 while reducing the coefficient norm to about 0.94. Notice the important trade-off: the constrained model is less free. It may fit the training observations slightly less closely, but its behavior on new districts is much more stable.

We should not trust a conclusion based on one convenient random sample, so the third panel repeats the entire experiment with 40 different sets of 100 sales. The constrained model wins in every repetition in this run, and the median test MSE falls from about 1.04 to about 0.67. That is the central lesson: when data are scarce relative to the number of effects we are trying to estimate, additional freedom can become fragility.

The code uses ridge regression, which is usually written as minimizing the MSE plus $\lambda\lVert w\rVert_2^2$. For an appropriate pairing of $\lambda$ and $R$, that penalized problem and the hard constraint $\lVert w\rVert_2\le R$ produce the same solution. The penalty and the constraint are two ways of expressing the same preference. The parameter $\lambda$ is the price of coefficient magnitude; the radius $R$ is a budget for it.

One caution about the shaded regions: values outside $[0,5]$ are outside the range recorded in this dataset. The upper endpoint exists because the original target was capped near $500,000; it is not a statement that a real California home cannot be worth more. This distinction matters. A useful constraint must encode something we genuinely know about the problem. A mistaken constraint can make a model confidently wrong.

So constraints are not merely technical obstacles added to an optimization problem. They are a language for expressing scarcity, stability, physical limits, or prior knowledge. Here the constraint says, “With only 100 sales, do not trust a complicated explanation that requires enormous coefficients.”

## Interactive Gradient Descent: One Model in Two Spaces

We have described training as minimizing a loss, but that description can still feel abstract. This interactive figure shows the same model simultaneously in two different spaces. The left panel is data space: each dot is a district, and the line is the current regression model. The right panel is parameter space: the horizontal coordinate is the weight, the vertical coordinate is the bias, and every possible point corresponds to one possible line on the left.

The key idea is that the red line and the red point are not two different objects. They are two views of exactly the same model. When I change the weight slider, the point moves horizontally on the right and the line rotates on the left. When I change the bias, the point moves vertically and the line shifts up or down.

*Begin with $w=0$, $b=0$, and a learning rate of $0.1$. Update the linked view.*

At $w=0$ and $b=0$, the model predicts the same standardized value for every district, so the red line is horizontal. The orange vertical segments are a sample of the residuals: the gaps between observed and predicted values. Squaring and averaging all of those gaps gives the MSE displayed in the title.

On the right, the contour lines connect parameter choices with equal loss. The center of the contours is the best-fitting combination of weight and bias. The orange arrow is the gradient. It points uphill, toward the direction in which the loss increases most rapidly. Gradient descent moves in the opposite direction, so the green update points downhill.

For the initial model, the weight derivative is approximately $-1.356$, while the bias derivative is essentially zero. Why is the bias gradient already zero? We standardized the target, so its mean is zero. A horizontal line at zero already has the best intercept for a centered target; what the model is missing is the relationship between income and house value. One step therefore changes the weight from zero to about $0.136$, while the bias barely moves. The loss falls from 1.00 to about 0.835.

This is also why standardization occurs immediately before this experiment. It is not an isolated result that students are meant to memorize. It places income and house value on comparable numerical scales, making one learning-rate control meaningful for both the weight and bias and keeping the parameter-space picture readable.

*Move the weight and bias sliders manually before pressing the update button.*

Try to improve the red line by eye. As we move through parameter space, we are selecting a new model; the dataset and the loss surface remain fixed. Some changes that look plausible on the left can still increase the total MSE because our eyes focus on a few visible points, whereas the objective averages all observations. This is why manual search becomes unreliable even with only two parameters.

Now compare learning rates. With a very small learning rate, such as $0.001$, the green step points in the correct direction but barely moves. The algorithm is safe but inefficient. With a moderate learning rate, the loss falls substantially in one step. With a large learning rate, the update can jump across the valley and may increase the loss. The gradient chooses a direction; the learning rate decides how much confidence to place in that local direction.

*Turn on “Show optimum.”*

The star marks the global minimum. For this linear regression problem with MSE, the loss surface is convex, so there is only one basin and one global solution. Gradient descent does not know the star's location in advance. It repeatedly measures the local slope and takes a step. The optimization trajectory emerges from a sequence of simple local decisions.

The two panels give us the bridge we will use for the rest of the lecture. In data space, training changes predictions and residuals. In parameter space, the same training process moves a point downhill on a loss surface. Gradient descent is the rule that keeps those two views synchronized.
