<!-- classes: title -->

<!-- note
Hello everyone,

In this talk I'm going to show you how to design functions that can be correctly graph-converted using two of the most exciting new features of TensorFlow 2.0: AutoGraph and tf.function.


But first, let me introduce myself.

-->

# Dissecting tf.function to discover AutoGraph strengths and subtleties

<small><i>How to correctly write graph-convertible functions in TensorFlow 2.0.</i></small>

---

## About me

<!-- note
I'm Paolo Galeone, I'm a computer engineer, I do Computer Vision and Machine Learning for a living and... I'm obsessed with TensorFlow.

I started using TensorFlow as soon as Google released it publicly, around November 2015, when I was a Research fellow at the computer vision laboratory of the university of Bologna,

And I never stopped since then.

In fact, I blog about TensorFlow, I answer question on StackOverflow about TensorFlow, I write opensource software with TensorFlow, and I use it everyday at work.

Google noticed my passion and awarded me with the title of Google Developer Expert in Machine Learning.

This also allowed me to be here today, thanks to their travel support program :)

As I mentioned, I have a blog (**point to the address**) in which I write about TensorFlow mainly and I invite you to go read it, especially because this talk is born from a three part article I wrote about tf.function and autograph.

Ok, we are ready to start!

-->
![me](https://pbs.twimg.com/profile_images/949951677402140672/boueakkR_400x400.jpg)
<!-- TODO: better pic -->

Computer engineer | Head of ML & CV @ ZURU Tech Italy | Machine Learning GDE

- Blog: https://pgaleone.eu/
- Github: [https://github.com/galeone/](galeone)
- Twitter: [@paolo_galeone](https://twitter.com/paolo_galeone)

---

<!-- sectionTitle: DataFlow Graphs & TensorFlow 2.0 -->
<!-- note

In TF 2.0 the concept of graph definition and session execution, core of the descriptive way of programming used in TF 1.x, are disappeared, or better they have been hidden, if favor of the eager execution.

The eager execution is just the Python-like execution of the computation, line by line.

These new design choice has been made with the aim of lowering the entry barriers, making TensorFlow more pythonic and easy to use.

-->

## DataFlow Graphs & TensorFlow 2.0

---

## Graphs Advantages

<!-- note

Of course, the description of the computation using dataflow graphs, proper of TensorFlow 1.x, have too many advantages that TensorFlow 2.0 must have, like:

- a faster execution speed, graphs are easy to replicate and distribute, and
- automatic differentiation is implemented describing the conputation using graphs
- ...

TF 2.0 brings together the ease of eager execution and the power of TF 1, and 
at the center of this merger there is tf.function.
-->

- Execution Speed
- Language Agnostic Representation
- Easy to replicate and distribute
- Automatic Differentiation

---

<!-- note
tf.function allows you to transform a subset of Python syntax into portable, high-performance TensorFlow graph 1.x like, with a simple function decoration.

As it can be seen from the function signature, tf.function uses autograph.

AutoGraph lets you write graph code using natural Python syntax. In particular, AutoGraph allows us to use Python control flow statements inside tf.function decorated functions, and it automatically converts them into the appropriate TensorFlow graph ops.

-->

# tf.function and AutoGraph

```python
# tf.function signature: it is a decorator.
def function(func=None,
             input_signature=None,
             autograph=True,
             experimental_autograph_options=None)
```

<b>tf.function uses AutoGraph</b>

AutoGraph converts Python control flow statements into appropriate TensorFlow graph ops.

---

# tf.function and AutoGraph

<!-- note

TODO: description of the diagram

Since tf.function is a decorator, it is needed to organize the code using functions, that have been replaced the TensorFlow 1.x sessions.

For instance, given a problem to solve
-->

![tf-execution](images/tf-execution.png)

---

<!-- note
That is the multiplication of 2 constant matrix followed by the addition of a scalar variable,
-->

## Program to Solve

Given the **constant** matrices
<br />

![aX](/images/Ax.png)
<br />

And the scalar **variable** ![b](/images/b.png)
<br />

Compute ![Ax_b](/images/Ax_b.png)

---

## TensorFlow 1.x solution

<!-- note
In TensorFlow 1.x we have to
- first **describe the computation** as a **graph**
- then create a special node, with the only goal of initializing the variables
- then create a session object, that is the object that receives the description of the coputation and places it on the correct hardware
- and finally use the session object to run the computation and getting the result

in TensorFlow 2.0, thanks to the eager execution the solution of the problem becomes easier.
-->

```python
g = tf.Graph()
with g.as_default():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y))
```

---

## TensorFlow 2.0 solution: eager execution

<!-- note
In fact, we only have to declare the constants and the variables, and the computation is executed directly, without the need to create a session.

In order to replicate the same behaviour of the session execution, we just write the code inside a function.

Executing the function has the same behaviour of the previous session.run.

The only peculiarity here, is that every `tf` operation, like tf.constant, tf.matmul and so on, produce a `tf.Tensor` object and not a Python native type or a numpy array.

Therefore, we have to extract from the tf.Tensor the Numpy array representation by calling the numpy method of the `tf.Tensor` object.

 We can call the function as many times we want, and it works like any other Python function.

Alright then, since we declared the computation inside a function, we can try to convert it to its graph representation using tf.function.
-->

```python
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    return y
print([f().numpy() for _ in range(10)])
```

<b>Every tf.* op, produces a `tf.Tensor` object</b>

---

## From eager function to tf.function

<!-- note
One might expect that since this function works in eager mode, we can convert it to its graph representation only by decorating it with tf.function.

Let's try and see what happens - I added a print statement and a tf.print statement that will help to clarify what happens here.
-->

```python
@tf.function
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

f()
```

---

<!-- note
OK, there is some output on the console.
When the function f() is called he process of graph building starts. At this stage, only the Python code is executed and the behavior of the function is traced, in order to collect the required data to build the graph.

The tf.print call is not evaluated as any other tf.* method, since Tensorflow already knows everything about that statements and it can use them as they are to build the graph.
-->

## From eager function to tf.function

```text
PRINT:  Tensor("add:0", shape=(2, 2), dtype=float32)
```

---

## From eager function to tf.function

```text
ValueError: tf.function-decorated function tried to create variables on non-first call.
```


---

## Lesson #1

> Converting a function that works in eager mode to its graph representation requires to think about the graph even though we are working in eager mode.

