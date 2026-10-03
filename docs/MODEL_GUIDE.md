# Model Guide: M1, M2, and M3

This document explains the three models in this project in lecturer-level
detail. The current implementation uses PyTorch for all three models. M3 uses
transfer learning from a pretrained MobileNetV2 model.

## 1. Common task and data flow

The task is 28-class image classification:

- 14 fruit or vegetable types
- 2 conditions: Healthy and Rotten
- One output class represents a pair such as `Apple__Healthy` or
  `Apple__Rotten`

The common pipeline is:

1. Read and validate image files.
2. Remove invalid images and exclude conflicting duplicate hashes.
3. Keep identical images in the same split using SHA-256 hashes.
4. Split the data into 70% training, 15% validation, and 15% test data.
5. Resize each image while preserving its aspect ratio with padding.
6. Apply ImageNet normalization:

   ```text
   normalized = (pixel / 255 - mean) / standard_deviation
   ```

7. Train using weighted cross-entropy loss.
8. Select the best checkpoint using validation macro-F1.
9. Evaluate on the test set using accuracy, macro precision, macro recall,
   macro-F1, and a confusion matrix.

The class-weighted loss gives more influence to classes with fewer training
images:

```text
weight(class) = number_of_training_images /
                (number_of_classes * images_in_class)
```

## 2. M1: Simple fully connected neural network

### Purpose

M1 is the baseline. It tests how well a standard multilayer perceptron can
classify images when the two-dimensional spatial structure is discarded.

### Input

Images are resized to `128 x 128` pixels with 3 RGB channels.

```text
Input shape:  (batch, 3, 128, 128)
Flattened:    (batch, 49,152)
```

The flattened vector contains all red, green, and blue values one after
another. A dense layer does not know that neighboring pixels are neighbors.

### Architecture

| Stage | Operation | Output shape |
|---|---|---:|
| 1 | Flatten | `(batch, 49,152)` |
| 2 | Linear `49,152 -> 256` | `(batch, 256)` |
| 3 | ReLU | `(batch, 256)` |
| 4 | Dropout `p=0.3` during training | `(batch, 256)` |
| 5 | Linear `256 -> 128` | `(batch, 128)` |
| 6 | ReLU | `(batch, 128)` |
| 7 | Dropout `p=0.3` during training | `(batch, 128)` |
| 8 | Linear `128 -> 28` | `(batch, 28)` |

The final 28 values are logits. They are not probabilities until passed
through softmax, which is handled internally by cross-entropy loss.

### Forward pass

For a dense layer:

```text
z = xW + b
```

ReLU is:

```text
ReLU(z) = max(0, z)
```

The final logits are converted to class probabilities conceptually by:

```text
softmax(z_i) = exp(z_i) / sum(exp(z_j))
```

### Why M1 is useful

- Simple baseline for comparison.
- Fast to understand and implement.
- Shows the cost of ignoring image locality.

### Main weakness

The first layer alone has approximately 12.6 million weights:

```text
49,152 * 256 + 256 = 12,583,168 parameters
```

This is large for a baseline and does not exploit repeated visual patterns.
For example, the same edge detector would need to be learned separately at
many pixel locations.

## 3. M2: Deep convolutional neural network

### Purpose

M2 improves on M1 by preserving spatial structure and learning local visual
features such as edges, color transitions, spots, texture, and shapes.

### Input

```text
Input shape: (batch, 3, 128, 128)
```

### Block-by-block architecture

| Block | Operations | Output shape |
|---|---|---:|
| Block 1 | Conv `3 -> 32`, kernel `3x3`, padding 1; BatchNorm; ReLU; MaxPool `2x2` | `(batch, 32, 64, 64)` |
| Block 2 | Conv `32 -> 64`, kernel `3x3`, padding 1; BatchNorm; ReLU; MaxPool `2x2` | `(batch, 64, 32, 32)` |
| Block 3 | Conv `64 -> 128`, kernel `3x3`, padding 1; BatchNorm; ReLU; MaxPool `2x2` | `(batch, 128, 16, 16)` |
| Head | Adaptive average pooling to `1x1` | `(batch, 128, 1, 1)` |
| Classifier | Flatten; Dropout `p=0.5`; Linear `128 -> 28` | `(batch, 28)` |

### What each operation does

#### Convolution

A convolution applies a small learned filter across the image. With a
`3x3` kernel and padding 1, the height and width stay unchanged before
pooling. Each filter detects a different pattern.

The first block can learn simple patterns such as:

- Horizontal and vertical edges
- Color boundaries
- Bright or dark regions

Later blocks combine earlier patterns into:

- Curves and texture
- Fruit outlines
- Bruising and surface defects
- Shape and freshness-related visual structure

#### Batch normalization

Batch normalization normalizes intermediate activations and then learns a
scale and shift. It usually makes optimization more stable and can allow
faster training.

#### ReLU

ReLU introduces non-linearity:

```text
ReLU(x) = max(0, x)
```

Without non-linear activations, multiple linear layers would still behave like
one linear transformation.

#### Max pooling

`2x2` max pooling keeps the strongest activation in each local region. It
reduces spatial dimensions by half:

```text
128 -> 64 -> 32 -> 16
```

This reduces computation and gives limited translation tolerance.

#### Adaptive average pooling

Adaptive average pooling averages each of the 128 feature maps across its
spatial positions. It converts `(128, 16, 16)` into `(128, 1, 1)` regardless
of the exact preceding spatial size.

This avoids a large fully connected layer and reduces overfitting.

#### Dropout

The classifier dropout randomly removes 50% of classifier inputs during
training. This regularizes the final decision layer. Dropout is disabled
during validation and testing.

### Why M2 should outperform M1

M2 has local connectivity and weight sharing. One filter can detect a useful
pattern at many positions. Pooling reduces irrelevant location details, while
deeper blocks build increasingly meaningful features.

M2 is still trained from random initialization, so it normally needs more
data and training time than a pretrained model.

## 4. M3: MobileNetV2 transfer learning

### Purpose

M3 starts with visual features learned from a large external dataset instead
of learning every feature from random initialization. This is transfer
learning.

### Architecture concept

MobileNetV2 is a lightweight CNN built from inverted residual blocks. Its
important ideas are:

1. Expand channels with a `1x1` convolution.
2. Apply a depthwise spatial convolution.
3. Project channels back down with another `1x1` convolution.
4. Add a residual connection when input and output shapes allow it.

Depthwise convolution applies one spatial filter per channel, which is much
cheaper than a full convolution across all input and output channels.

The current model replaces MobileNetV2's original classifier with:

```text
Linear(previous_feature_count, 28)
```

The output is therefore adapted from the original pretrained task to this
project's 28 classes.

### M3 training stages

#### Stage 1: Frozen backbone

For the first five epochs:

- MobileNetV2 feature layers are frozen.
- Their pretrained weights do not receive gradients.
- Only the new 28-class classifier is trained.

This teaches the new classifier how to map existing visual features to fruit
and vegetable classes.

#### Stage 2: Fine-tuning

After the frozen stage:

- Most or all of the backbone is unfrozen according to the project function.
- The optimizer is recreated with a smaller learning rate.
- The pretrained features are gently adapted to freshness classification.

The smaller learning rate protects useful pretrained representations from
being destroyed by large updates.

### Why transfer learning helps

The pretrained network already understands general visual primitives such as
edges, colors, textures, and shapes. The project only needs to adapt these
features to the new classes. This is especially useful when the project
dataset is smaller than the dataset used to pretrain the network.

### Limitations and lecturer points

- Pretrained weights come from a different dataset and task.
- M3 is not learning from zero.
- The input preprocessing should be compatible with the pretrained weights.
- Very high test performance should still be checked for dataset bias,
  duplicates, or visually easy examples.
- Fine-tuning too aggressively can cause catastrophic forgetting.

## 5. Training objective and evaluation

For a target class `y` and logits `z`, cross-entropy is:

```text
loss = -log(softmax(z)[y])
```

Training uses Adam to update learnable parameters from gradients:

```text
parameter <- parameter - learning_rate * Adam(gradient)
```

Accuracy measures the fraction of correct predictions. Macro precision,
recall, and F1 calculate the metric per class and then average classes equally.
Macro-F1 is important here because class frequencies are not identical.

A confusion matrix shows which true classes are confused with which predicted
classes. Off-diagonal cells identify specific weaknesses.

## 6. Short comparison

| Property | M1 | M2 | M3 |
|---|---|---|---|
| Core idea | Dense baseline | Learned spatial features | Reuse pretrained features |
| Initialization | Random | Random | Pretrained backbone + new head |
| Spatial awareness | No | Yes | Yes |
| Training cost | High parameter count | Moderate | Higher setup, usually efficient convergence |
| Expected performance | Lowest baseline | Better than M1 | Usually highest |
| Main teaching point | Dense classification baseline | CNN feature extraction | Transfer learning and fine-tuning |
