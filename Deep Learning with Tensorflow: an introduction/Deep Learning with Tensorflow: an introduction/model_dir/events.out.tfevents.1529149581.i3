       ЃK"	  @#>ЩжAbrain.Event:2пpњІБ      {f	яs|#>ЩжA"т

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 

global_step
VariableV2*
shared_name *
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 

!global_step/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_output_shapes
: *
_class
loc:@global_step

global_step/cond/SwitchSwitch!global_step/IsVariableInitialized!global_step/IsVariableInitialized*
_output_shapes
: : *
T0

a
global_step/cond/switch_tIdentityglobal_step/cond/Switch:1*
T0
*
_output_shapes
: 
_
global_step/cond/switch_fIdentityglobal_step/cond/Switch*
T0
*
_output_shapes
: 
h
global_step/cond/pred_idIdentity!global_step/IsVariableInitialized*
_output_shapes
: *
T0

b
global_step/cond/readIdentityglobal_step/cond/read/Switch:1*
T0	*
_output_shapes
: 

global_step/cond/read/Switch	RefSwitchglobal_stepglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 

global_step/cond/Switch_1Switchglobal_step/Initializer/zerosglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 
}
global_step/cond/MergeMergeglobal_step/cond/Switch_1global_step/cond/read*
N*
_output_shapes
: : *
T0	
S
global_step/add/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
b
global_step/addAddglobal_step/cond/Mergeglobal_step/add/y*
_output_shapes
: *
T0	

MatchingFiles/patternConst"/device:CPU:0*
dtype0*
_output_shapes
: *.
value%B# B/data/dogscats/train/**/*.jpg
i
MatchingFilesMatchingFilesMatchingFiles/pattern"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ
a
ShapeShapeMatchingFiles"/device:CPU:0*
_output_shapes
:*
T0*
out_type0	
l
strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:
n
strided_slice/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
n
strided_slice/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2"/device:CPU:0*
end_mask *
_output_shapes
: *
T0	*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
Z
	Maximum/yConst"/device:CPU:0*
value	B	 R*
dtype0	*
_output_shapes
: 
\
MaximumMaximumstrided_slice	Maximum/y"/device:CPU:0*
T0	*
_output_shapes
: 
U
seedConst"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R 
V
seed2Const"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R 
]
buffer_sizeConst"/device:CPU:0*
dtype0	*
_output_shapes
: *
value
B	 R 
V
countConst"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R

W
seed_1Const"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
X
seed2_1Const"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R 
[

batch_sizeConst"/device:CPU:0*
value	B	 R@*
dtype0	*
_output_shapes
: 
_
buffer_size_1Const"/device:CPU:0*
dtype0	*
_output_shapes
: *
value
B	 R 
і
OneShotIteratorOneShotIterator"/device:CPU:0*
_output_shapes
: *0
dataset_factoryR
_make_dataset_GUdGPfB2wf0*
shared_name *=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*
	container *
output_types
2
h
IteratorToStringHandleIteratorToStringHandleOneShotIterator"/device:CPU:0*
_output_shapes
: 
й
IteratorGetNextIteratorGetNextOneShotIterator"/device:CPU:0*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*>
_output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*
output_types
2
w
layer_1/truncated_normal/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
b
layer_1/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
layer_1/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
В
(layer_1/truncated_normal/TruncatedNormalTruncatedNormallayer_1/truncated_normal/shape*
T0*
dtype0*&
_output_shapes
: *
seed2 *

seed 

layer_1/truncated_normal/mulMul(layer_1/truncated_normal/TruncatedNormallayer_1/truncated_normal/stddev*&
_output_shapes
: *
T0

layer_1/truncated_normalAddlayer_1/truncated_normal/mullayer_1/truncated_normal/mean*&
_output_shapes
: *
T0

layer_1/Variable
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
Ь
layer_1/Variable/AssignAssignlayer_1/Variablelayer_1/truncated_normal*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*#
_class
loc:@layer_1/Variable

layer_1/Variable/readIdentitylayer_1/Variable*
T0*#
_class
loc:@layer_1/Variable*&
_output_shapes
: 
Z
layer_1/ConstConst*
valueB *ЭЬЬ=*
dtype0*
_output_shapes
: 
~
layer_1/Variable_1
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Л
layer_1/Variable_1/AssignAssignlayer_1/Variable_1layer_1/Const*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@layer_1/Variable_1

layer_1/Variable_1/readIdentitylayer_1/Variable_1*
_output_shapes
: *
T0*%
_class
loc:@layer_1/Variable_1
ш
layer_1/Conv2DConv2DIteratorGetNextlayer_1/Variable/read*/
_output_shapes
:џџџџџџџџџ *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
u
layer_1/addAddlayer_1/Conv2Dlayer_1/Variable_1/read*
T0*/
_output_shapes
:џџџџџџџџџ 
[
layer_1/ReluRelulayer_1/add*/
_output_shapes
:џџџџџџџџџ *
T0
Д
layer_1/MaxPoolMaxPoollayer_1/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
T0
w
layer_2/truncated_normal/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
b
layer_2/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
d
layer_2/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=
В
(layer_2/truncated_normal/TruncatedNormalTruncatedNormallayer_2/truncated_normal/shape*
dtype0*&
_output_shapes
: @*
seed2 *

seed *
T0

layer_2/truncated_normal/mulMul(layer_2/truncated_normal/TruncatedNormallayer_2/truncated_normal/stddev*
T0*&
_output_shapes
: @

layer_2/truncated_normalAddlayer_2/truncated_normal/mullayer_2/truncated_normal/mean*&
_output_shapes
: @*
T0

layer_2/Variable
VariableV2*
shared_name *
dtype0*&
_output_shapes
: @*
	container *
shape: @
Ь
layer_2/Variable/AssignAssignlayer_2/Variablelayer_2/truncated_normal*
use_locking(*
T0*#
_class
loc:@layer_2/Variable*
validate_shape(*&
_output_shapes
: @

layer_2/Variable/readIdentitylayer_2/Variable*
T0*#
_class
loc:@layer_2/Variable*&
_output_shapes
: @
Z
layer_2/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*ЭЬЬ=
~
layer_2/Variable_1
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Л
layer_2/Variable_1/AssignAssignlayer_2/Variable_1layer_2/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@layer_2/Variable_1

layer_2/Variable_1/readIdentitylayer_2/Variable_1*
T0*%
_class
loc:@layer_2/Variable_1*
_output_shapes
:@
ш
layer_2/Conv2DConv2Dlayer_1/MaxPoollayer_2/Variable/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
u
layer_2/addAddlayer_2/Conv2Dlayer_2/Variable_1/read*/
_output_shapes
:џџџџџџџџџ@*
T0
[
layer_2/ReluRelulayer_2/add*/
_output_shapes
:џџџџџџџџџ@*
T0
Д
layer_2/MaxPoolMaxPoollayer_2/Relu*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
T0*
data_formatNHWC*
strides

q
 fc_layer1/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"@     
d
fc_layer1/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
f
!fc_layer1/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=
А
*fc_layer1/truncated_normal/TruncatedNormalTruncatedNormal fc_layer1/truncated_normal/shape*
dtype0* 
_output_shapes
:
Р*
seed2 *

seed *
T0

fc_layer1/truncated_normal/mulMul*fc_layer1/truncated_normal/TruncatedNormal!fc_layer1/truncated_normal/stddev* 
_output_shapes
:
Р*
T0

fc_layer1/truncated_normalAddfc_layer1/truncated_normal/mulfc_layer1/truncated_normal/mean* 
_output_shapes
:
Р*
T0

fc_layer1/Variable
VariableV2*
dtype0* 
_output_shapes
:
Р*
	container *
shape:
Р*
shared_name 
Ю
fc_layer1/Variable/AssignAssignfc_layer1/Variablefc_layer1/truncated_normal*
use_locking(*
T0*%
_class
loc:@fc_layer1/Variable*
validate_shape(* 
_output_shapes
:
Р

fc_layer1/Variable/readIdentityfc_layer1/Variable* 
_output_shapes
:
Р*
T0*%
_class
loc:@fc_layer1/Variable
^
fc_layer1/ConstConst*
valueB*ЭЬЬ=*
dtype0*
_output_shapes	
:

fc_layer1/Variable_1
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ф
fc_layer1/Variable_1/AssignAssignfc_layer1/Variable_1fc_layer1/Const*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@fc_layer1/Variable_1

fc_layer1/Variable_1/readIdentityfc_layer1/Variable_1*
T0*'
_class
loc:@fc_layer1/Variable_1*
_output_shapes	
:
h
fc_layer1/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ@  

fc_layer1/ReshapeReshapelayer_2/MaxPoolfc_layer1/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџР

fc_layer1/MatMulMatMulfc_layer1/Reshapefc_layer1/Variable/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
t
fc_layer1/addAddfc_layer1/MatMulfc_layer1/Variable_1/read*(
_output_shapes
:џџџџџџџџџ*
T0
X
fc_layer1/ReluRelufc_layer1/add*(
_output_shapes
:џџџџџџџџџ*
T0
t
#read_out_fc2/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
g
"read_out_fc2/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$read_out_fc2/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=
Е
-read_out_fc2/truncated_normal/TruncatedNormalTruncatedNormal#read_out_fc2/truncated_normal/shape*
dtype0*
_output_shapes
:	
*
seed2 *

seed *
T0
Ї
!read_out_fc2/truncated_normal/mulMul-read_out_fc2/truncated_normal/TruncatedNormal$read_out_fc2/truncated_normal/stddev*
_output_shapes
:	
*
T0

read_out_fc2/truncated_normalAdd!read_out_fc2/truncated_normal/mul"read_out_fc2/truncated_normal/mean*
T0*
_output_shapes
:	


read_out_fc2/Variable
VariableV2*
shared_name *
dtype0*
_output_shapes
:	
*
	container *
shape:	

й
read_out_fc2/Variable/AssignAssignread_out_fc2/Variableread_out_fc2/truncated_normal*
validate_shape(*
_output_shapes
:	
*
use_locking(*
T0*(
_class
loc:@read_out_fc2/Variable

read_out_fc2/Variable/readIdentityread_out_fc2/Variable*
T0*(
_class
loc:@read_out_fc2/Variable*
_output_shapes
:	

_
read_out_fc2/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*ЭЬЬ=

read_out_fc2/Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes
:
*
	container *
shape:

Я
read_out_fc2/Variable_1/AssignAssignread_out_fc2/Variable_1read_out_fc2/Const*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0**
_class 
loc:@read_out_fc2/Variable_1

read_out_fc2/Variable_1/readIdentityread_out_fc2/Variable_1*
_output_shapes
:
*
T0**
_class 
loc:@read_out_fc2/Variable_1
Ё
read_out_fc2/MatMulMatMulfc_layer1/Reluread_out_fc2/Variable/read*
T0*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
transpose_b( 
|
read_out_fc2/addAddread_out_fc2/MatMulread_out_fc2/Variable_1/read*
T0*'
_output_shapes
:џџџџџџџџџ

z
)SparseSoftmaxCrossEntropyWithLogits/ShapeShapeIteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
у
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsread_out_fc2/addIteratorGetNext:1*
T0*6
_output_shapes$
":џџџџџџџџџ:џџџџџџџџџ
*
Tlabels0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:

MeanMeanGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMaxArgMaxread_out_fc2/addArgMax/dimension*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
Q
CastCastArgMax*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
U
EqualEqualCastIteratorGetNext:1*
T0*#
_output_shapes
:џџџџџџџџџ
S
ToFloatCastEqual*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0


acc_op/total/Initializer/zerosConst*
_class
loc:@acc_op/total*
valueB
 *    *
dtype0*
_output_shapes
: 

acc_op/total
VariableV2*
shared_name *
_class
loc:@acc_op/total*
	container *
shape: *
dtype0*
_output_shapes
: 
Ж
acc_op/total/AssignAssignacc_op/totalacc_op/total/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@acc_op/total
m
acc_op/total/readIdentityacc_op/total*
T0*
_class
loc:@acc_op/total*
_output_shapes
: 

acc_op/count/Initializer/zerosConst*
_class
loc:@acc_op/count*
valueB
 *    *
dtype0*
_output_shapes
: 

acc_op/count
VariableV2*
shared_name *
_class
loc:@acc_op/count*
	container *
shape: *
dtype0*
_output_shapes
: 
Ж
acc_op/count/AssignAssignacc_op/countacc_op/count/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@acc_op/count
m
acc_op/count/readIdentityacc_op/count*
T0*
_class
loc:@acc_op/count*
_output_shapes
: 
M
acc_op/SizeSizeToFloat*
_output_shapes
: *
T0*
out_type0
U
acc_op/ToFloat_1Castacc_op/Size*

SrcT0*
_output_shapes
: *

DstT0
V
acc_op/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
f

acc_op/SumSumToFloatacc_op/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

acc_op/AssignAdd	AssignAddacc_op/total
acc_op/Sum*
use_locking( *
T0*
_class
loc:@acc_op/total*
_output_shapes
: 

acc_op/AssignAdd_1	AssignAddacc_op/countacc_op/ToFloat_1^ToFloat*
use_locking( *
T0*
_class
loc:@acc_op/count*
_output_shapes
: 
`
acc_op/truedivRealDivacc_op/total/readacc_op/count/read*
T0*
_output_shapes
: 
V
acc_op/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
`
acc_op/GreaterGreateracc_op/count/readacc_op/zeros_like*
T0*
_output_shapes
: 
j
acc_op/valueSelectacc_op/Greateracc_op/truedivacc_op/zeros_like*
T0*
_output_shapes
: 
b
acc_op/truediv_1RealDivacc_op/AssignAddacc_op/AssignAdd_1*
T0*
_output_shapes
: 
X
acc_op/zeros_like_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
e
acc_op/Greater_1Greateracc_op/AssignAdd_1acc_op/zeros_like_1*
T0*
_output_shapes
: 
t
acc_op/update_opSelectacc_op/Greater_1acc_op/truediv_1acc_op/zeros_like_1*
T0*
_output_shapes
: 
V
accuracy/tagsConst*
dtype0*
_output_shapes
: *
valueB Baccuracy
[
accuracyScalarSummaryaccuracy/tagsacc_op/update_op*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
 
gradients/Mean_grad/ShapeShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
Ђ
gradients/Mean_grad/Shape_1ShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:џџџџџџџџџ

­
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:џџџџџџџџџ
*Д
messageЈЅCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
T0
А
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Б
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Mean_grad/truedivegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
о
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*'
_output_shapes
:џџџџџџџџџ
*
T0
x
%gradients/read_out_fc2/add_grad/ShapeShaperead_out_fc2/MatMul*
_output_shapes
:*
T0*
out_type0
q
'gradients/read_out_fc2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:

л
5gradients/read_out_fc2/add_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/read_out_fc2/add_grad/Shape'gradients/read_out_fc2/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
§
#gradients/read_out_fc2/add_grad/SumSumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul5gradients/read_out_fc2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
О
'gradients/read_out_fc2/add_grad/ReshapeReshape#gradients/read_out_fc2/add_grad/Sum%gradients/read_out_fc2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ


%gradients/read_out_fc2/add_grad/Sum_1SumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul7gradients/read_out_fc2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
)gradients/read_out_fc2/add_grad/Reshape_1Reshape%gradients/read_out_fc2/add_grad/Sum_1'gradients/read_out_fc2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:


0gradients/read_out_fc2/add_grad/tuple/group_depsNoOp(^gradients/read_out_fc2/add_grad/Reshape*^gradients/read_out_fc2/add_grad/Reshape_1

8gradients/read_out_fc2/add_grad/tuple/control_dependencyIdentity'gradients/read_out_fc2/add_grad/Reshape1^gradients/read_out_fc2/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/read_out_fc2/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ


:gradients/read_out_fc2/add_grad/tuple/control_dependency_1Identity)gradients/read_out_fc2/add_grad/Reshape_11^gradients/read_out_fc2/add_grad/tuple/group_deps*
_output_shapes
:
*
T0*<
_class2
0.loc:@gradients/read_out_fc2/add_grad/Reshape_1
т
)gradients/read_out_fc2/MatMul_grad/MatMulMatMul8gradients/read_out_fc2/add_grad/tuple/control_dependencyread_out_fc2/Variable/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(
Я
+gradients/read_out_fc2/MatMul_grad/MatMul_1MatMulfc_layer1/Relu8gradients/read_out_fc2/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	
*
transpose_a(*
transpose_b( 

3gradients/read_out_fc2/MatMul_grad/tuple/group_depsNoOp*^gradients/read_out_fc2/MatMul_grad/MatMul,^gradients/read_out_fc2/MatMul_grad/MatMul_1

;gradients/read_out_fc2/MatMul_grad/tuple/control_dependencyIdentity)gradients/read_out_fc2/MatMul_grad/MatMul4^gradients/read_out_fc2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/read_out_fc2/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

=gradients/read_out_fc2/MatMul_grad/tuple/control_dependency_1Identity+gradients/read_out_fc2/MatMul_grad/MatMul_14^gradients/read_out_fc2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/read_out_fc2/MatMul_grad/MatMul_1*
_output_shapes
:	

В
&gradients/fc_layer1/Relu_grad/ReluGradReluGrad;gradients/read_out_fc2/MatMul_grad/tuple/control_dependencyfc_layer1/Relu*
T0*(
_output_shapes
:џџџџџџџџџ
r
"gradients/fc_layer1/add_grad/ShapeShapefc_layer1/MatMul*
_output_shapes
:*
T0*
out_type0
o
$gradients/fc_layer1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
в
2gradients/fc_layer1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/fc_layer1/add_grad/Shape$gradients/fc_layer1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
У
 gradients/fc_layer1/add_grad/SumSum&gradients/fc_layer1/Relu_grad/ReluGrad2gradients/fc_layer1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
$gradients/fc_layer1/add_grad/ReshapeReshape gradients/fc_layer1/add_grad/Sum"gradients/fc_layer1/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџ
Ч
"gradients/fc_layer1/add_grad/Sum_1Sum&gradients/fc_layer1/Relu_grad/ReluGrad4gradients/fc_layer1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Џ
&gradients/fc_layer1/add_grad/Reshape_1Reshape"gradients/fc_layer1/add_grad/Sum_1$gradients/fc_layer1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes	
:

-gradients/fc_layer1/add_grad/tuple/group_depsNoOp%^gradients/fc_layer1/add_grad/Reshape'^gradients/fc_layer1/add_grad/Reshape_1

5gradients/fc_layer1/add_grad/tuple/control_dependencyIdentity$gradients/fc_layer1/add_grad/Reshape.^gradients/fc_layer1/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/fc_layer1/add_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ќ
7gradients/fc_layer1/add_grad/tuple/control_dependency_1Identity&gradients/fc_layer1/add_grad/Reshape_1.^gradients/fc_layer1/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/fc_layer1/add_grad/Reshape_1*
_output_shapes	
:
й
&gradients/fc_layer1/MatMul_grad/MatMulMatMul5gradients/fc_layer1/add_grad/tuple/control_dependencyfc_layer1/Variable/read*(
_output_shapes
:џџџџџџџџџР*
transpose_a( *
transpose_b(*
T0
Э
(gradients/fc_layer1/MatMul_grad/MatMul_1MatMulfc_layer1/Reshape5gradients/fc_layer1/add_grad/tuple/control_dependency*
T0* 
_output_shapes
:
Р*
transpose_a(*
transpose_b( 

0gradients/fc_layer1/MatMul_grad/tuple/group_depsNoOp'^gradients/fc_layer1/MatMul_grad/MatMul)^gradients/fc_layer1/MatMul_grad/MatMul_1

8gradients/fc_layer1/MatMul_grad/tuple/control_dependencyIdentity&gradients/fc_layer1/MatMul_grad/MatMul1^gradients/fc_layer1/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџР*
T0*9
_class/
-+loc:@gradients/fc_layer1/MatMul_grad/MatMul

:gradients/fc_layer1/MatMul_grad/tuple/control_dependency_1Identity(gradients/fc_layer1/MatMul_grad/MatMul_11^gradients/fc_layer1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/fc_layer1/MatMul_grad/MatMul_1* 
_output_shapes
:
Р
u
&gradients/fc_layer1/Reshape_grad/ShapeShapelayer_2/MaxPool*
_output_shapes
:*
T0*
out_type0
н
(gradients/fc_layer1/Reshape_grad/ReshapeReshape8gradients/fc_layer1/MatMul_grad/tuple/control_dependency&gradients/fc_layer1/Reshape_grad/Shape*/
_output_shapes
:џџџџџџџџџ@*
T0*
Tshape0

*gradients/layer_2/MaxPool_grad/MaxPoolGradMaxPoolGradlayer_2/Relulayer_2/MaxPool(gradients/fc_layer1/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@
Є
$gradients/layer_2/Relu_grad/ReluGradReluGrad*gradients/layer_2/MaxPool_grad/MaxPoolGradlayer_2/Relu*/
_output_shapes
:џџџџџџџџџ@*
T0
n
 gradients/layer_2/add_grad/ShapeShapelayer_2/Conv2D*
T0*
out_type0*
_output_shapes
:
l
"gradients/layer_2/add_grad/Shape_1Const*
valueB:@*
dtype0*
_output_shapes
:
Ь
0gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/layer_2/add_grad/Shape"gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
gradients/layer_2/add_grad/SumSum$gradients/layer_2/Relu_grad/ReluGrad0gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
З
"gradients/layer_2/add_grad/ReshapeReshapegradients/layer_2/add_grad/Sum gradients/layer_2/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ@
С
 gradients/layer_2/add_grad/Sum_1Sum$gradients/layer_2/Relu_grad/ReluGrad2gradients/layer_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ј
$gradients/layer_2/add_grad/Reshape_1Reshape gradients/layer_2/add_grad/Sum_1"gradients/layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@

+gradients/layer_2/add_grad/tuple/group_depsNoOp#^gradients/layer_2/add_grad/Reshape%^gradients/layer_2/add_grad/Reshape_1

3gradients/layer_2/add_grad/tuple/control_dependencyIdentity"gradients/layer_2/add_grad/Reshape,^gradients/layer_2/add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/layer_2/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџ@
ѓ
5gradients/layer_2/add_grad/tuple/control_dependency_1Identity$gradients/layer_2/add_grad/Reshape_1,^gradients/layer_2/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:@

$gradients/layer_2/Conv2D_grad/ShapeNShapeNlayer_1/MaxPoollayer_2/Variable/read*
N* 
_output_shapes
::*
T0*
out_type0
|
#gradients/layer_2/Conv2D_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"          @   
§
1gradients/layer_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/layer_2/Conv2D_grad/ShapeNlayer_2/Variable/read3gradients/layer_2/add_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0
д
2gradients/layer_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterlayer_1/MaxPool#gradients/layer_2/Conv2D_grad/Const3gradients/layer_2/add_grad/tuple/control_dependency*&
_output_shapes
: @*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

.gradients/layer_2/Conv2D_grad/tuple/group_depsNoOp3^gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter2^gradients/layer_2/Conv2D_grad/Conv2DBackpropInput
І
6gradients/layer_2/Conv2D_grad/tuple/control_dependencyIdentity1gradients/layer_2/Conv2D_grad/Conv2DBackpropInput/^gradients/layer_2/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/layer_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ 
Ё
8gradients/layer_2/Conv2D_grad/tuple/control_dependency_1Identity2gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter/^gradients/layer_2/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @

*gradients/layer_1/MaxPool_grad/MaxPoolGradMaxPoolGradlayer_1/Relulayer_1/MaxPool6gradients/layer_2/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
T0
Є
$gradients/layer_1/Relu_grad/ReluGradReluGrad*gradients/layer_1/MaxPool_grad/MaxPoolGradlayer_1/Relu*
T0*/
_output_shapes
:џџџџџџџџџ 
n
 gradients/layer_1/add_grad/ShapeShapelayer_1/Conv2D*
_output_shapes
:*
T0*
out_type0
l
"gradients/layer_1/add_grad/Shape_1Const*
valueB: *
dtype0*
_output_shapes
:
Ь
0gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/layer_1/add_grad/Shape"gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
gradients/layer_1/add_grad/SumSum$gradients/layer_1/Relu_grad/ReluGrad0gradients/layer_1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
З
"gradients/layer_1/add_grad/ReshapeReshapegradients/layer_1/add_grad/Sum gradients/layer_1/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ 
С
 gradients/layer_1/add_grad/Sum_1Sum$gradients/layer_1/Relu_grad/ReluGrad2gradients/layer_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ј
$gradients/layer_1/add_grad/Reshape_1Reshape gradients/layer_1/add_grad/Sum_1"gradients/layer_1/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

+gradients/layer_1/add_grad/tuple/group_depsNoOp#^gradients/layer_1/add_grad/Reshape%^gradients/layer_1/add_grad/Reshape_1

3gradients/layer_1/add_grad/tuple/control_dependencyIdentity"gradients/layer_1/add_grad/Reshape,^gradients/layer_1/add_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *
T0*5
_class+
)'loc:@gradients/layer_1/add_grad/Reshape
ѓ
5gradients/layer_1/add_grad/tuple/control_dependency_1Identity$gradients/layer_1/add_grad/Reshape_1,^gradients/layer_1/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/layer_1/add_grad/Reshape_1*
_output_shapes
: 

$gradients/layer_1/Conv2D_grad/ShapeNShapeNIteratorGetNextlayer_1/Variable/read*
N* 
_output_shapes
::*
T0*
out_type0
|
#gradients/layer_1/Conv2D_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             
§
1gradients/layer_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/layer_1/Conv2D_grad/ShapeNlayer_1/Variable/read3gradients/layer_1/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д
2gradients/layer_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterIteratorGetNext#gradients/layer_1/Conv2D_grad/Const3gradients/layer_1/add_grad/tuple/control_dependency*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0

.gradients/layer_1/Conv2D_grad/tuple/group_depsNoOp3^gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter2^gradients/layer_1/Conv2D_grad/Conv2DBackpropInput
І
6gradients/layer_1/Conv2D_grad/tuple/control_dependencyIdentity1gradients/layer_1/Conv2D_grad/Conv2DBackpropInput/^gradients/layer_1/Conv2D_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*D
_class:
86loc:@gradients/layer_1/Conv2D_grad/Conv2DBackpropInput
Ё
8gradients/layer_1/Conv2D_grad/tuple/control_dependency_1Identity2gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter/^gradients/layer_1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
b
GradientDescent/learning_rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ј
<GradientDescent/update_layer_1/Variable/ApplyGradientDescentApplyGradientDescentlayer_1/VariableGradientDescent/learning_rate8gradients/layer_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_1/Variable*&
_output_shapes
: 

>GradientDescent/update_layer_1/Variable_1/ApplyGradientDescentApplyGradientDescentlayer_1/Variable_1GradientDescent/learning_rate5gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@layer_1/Variable_1*
_output_shapes
: 
Ј
<GradientDescent/update_layer_2/Variable/ApplyGradientDescentApplyGradientDescentlayer_2/VariableGradientDescent/learning_rate8gradients/layer_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_2/Variable*&
_output_shapes
: @

>GradientDescent/update_layer_2/Variable_1/ApplyGradientDescentApplyGradientDescentlayer_2/Variable_1GradientDescent/learning_rate5gradients/layer_2/add_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*%
_class
loc:@layer_2/Variable_1
Њ
>GradientDescent/update_fc_layer1/Variable/ApplyGradientDescentApplyGradientDescentfc_layer1/VariableGradientDescent/learning_rate:gradients/fc_layer1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@fc_layer1/Variable* 
_output_shapes
:
Р
Ј
@GradientDescent/update_fc_layer1/Variable_1/ApplyGradientDescentApplyGradientDescentfc_layer1/Variable_1GradientDescent/learning_rate7gradients/fc_layer1/add_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*'
_class
loc:@fc_layer1/Variable_1
Е
AGradientDescent/update_read_out_fc2/Variable/ApplyGradientDescentApplyGradientDescentread_out_fc2/VariableGradientDescent/learning_rate=gradients/read_out_fc2/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	
*
use_locking( *
T0*(
_class
loc:@read_out_fc2/Variable
Г
CGradientDescent/update_read_out_fc2/Variable_1/ApplyGradientDescentApplyGradientDescentread_out_fc2/Variable_1GradientDescent/learning_rate:gradients/read_out_fc2/add_grad/tuple/control_dependency_1*
_output_shapes
:
*
use_locking( *
T0**
_class 
loc:@read_out_fc2/Variable_1
Ќ
GradientDescent/updateNoOp?^GradientDescent/update_fc_layer1/Variable/ApplyGradientDescentA^GradientDescent/update_fc_layer1/Variable_1/ApplyGradientDescent=^GradientDescent/update_layer_1/Variable/ApplyGradientDescent?^GradientDescent/update_layer_1/Variable_1/ApplyGradientDescent=^GradientDescent/update_layer_2/Variable/ApplyGradientDescent?^GradientDescent/update_layer_2/Variable_1/ApplyGradientDescentB^GradientDescent/update_read_out_fc2/Variable/ApplyGradientDescentD^GradientDescent/update_read_out_fc2/Variable_1/ApplyGradientDescent

GradientDescent/valueConst^GradientDescent/update*
dtype0	*
_output_shapes
: *
_class
loc:@global_step*
value	B	 R

GradientDescent	AssignAddglobal_stepGradientDescent/value*
use_locking( *
T0	*
_class
loc:@global_step*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
_output_shapes
: *
T0У+
ю

tf_map_func_6yLLF6rv0aU
arg0
resize_images_squeeze

cond_merge25A wrapper for Defun that facilitates shape inference./
ConstConst*
value	B B/*
dtype02
packedPackarg0*
N*
T0*

axis M
StringSplitStringSplitpacked:output:0Const:output:0*

skip_empty(J
strided_slice/stackConst*
valueB:
ўџџџџџџџџ*
dtype0L
strided_slice/stack_1Const*
valueB:
џџџџџџџџџ*
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:
strided_sliceStridedSliceStringSplit:values:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index04
Equal/yConst*
valueB
 Bdogs*
dtype0A
EqualEqualstrided_slice:output:0Equal/y:output:0*
T04
cond/SwitchSwitch	Equal:z:0	Equal:z:0*
T0
=
cond/switch_tIdentitycond/Switch:output_true:0*
T0
>
cond/switch_fIdentitycond/Switch:output_false:0*
T0
,
cond/pred_idIdentity	Equal:z:0*
T0
D

cond/ConstConst^cond/switch_t*
value	B : *
dtype0F
cond/Const_1Const^cond/switch_f*
value	B :*
dtype0Q

cond/MergeMergecond/Const_1:output:0cond/Const:output:0*
T0*
N
ReadFileReadFilearg0Ў

DecodeJpeg
DecodeJpegReadFile:contents:0*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( F
convert_image/CastCastDecodeJpeg:image:0*

SrcT0*

DstT0<
convert_image/yConst*
valueB
 *;*
dtype0O
convert_imageMulconvert_image/Cast:y:0convert_image/y:output:0*
T0F
resize_images/ExpandDims/dimConst*
value	B : *
dtype0u
resize_images/ExpandDims
ExpandDimsconvert_image:z:0%resize_images/ExpandDims/dim:output:0*

Tdim0*
T0G
resize_images/sizeConst*
valueB"      *
dtype0
resize_images/ResizeBilinearResizeBilinear!resize_images/ExpandDims:output:0resize_images/size:output:0*
align_corners( *
T0o
resize_images/SqueezeSqueeze-resize_images/ResizeBilinear:resized_images:0*
squeeze_dims
 *
T0"!

cond_mergecond/Merge:output:0"7
resize_images_squeezeresize_images/Squeeze:output:0
Я
3
_make_dataset_GUdGPfB2wf0
prefetchdatasetn
(TensorSliceDataset/MatchingFiles/patternConst*
dtype0*.
value%B# B/data/dogscats/train/**/*.jpgd
 TensorSliceDataset/MatchingFilesMatchingFiles1TensorSliceDataset/MatchingFiles/pattern:output:0
TensorSliceDatasetTensorSliceDataset,TensorSliceDataset/MatchingFiles:filenames:0*
output_shapes
: *
Toutput_types
2j
$ShuffleDataset/MatchingFiles/patternConst*.
value%B# B/data/dogscats/train/**/*.jpg*
dtype0\
ShuffleDataset/MatchingFilesMatchingFiles-ShuffleDataset/MatchingFiles/pattern:output:0`
ShuffleDataset/ShapeShape(ShuffleDataset/MatchingFiles:filenames:0*
T0*
out_type0	P
"ShuffleDataset/strided_slice/stackConst*
dtype0*
valueB: R
$ShuffleDataset/strided_slice/stack_1Const*
dtype0*
valueB:R
$ShuffleDataset/strided_slice/stack_2Const*
dtype0*
valueB:а
ShuffleDataset/strided_sliceStridedSliceShuffleDataset/Shape:output:0+ShuffleDataset/strided_slice/stack:output:0-ShuffleDataset/strided_slice/stack_1:output:0-ShuffleDataset/strided_slice/stack_2:output:0*
end_mask *
Index0*
T0	*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask B
ShuffleDataset/Maximum/yConst*
value	B	 R*
dtype0	t
ShuffleDataset/MaximumMaximum%ShuffleDataset/strided_slice:output:0!ShuffleDataset/Maximum/y:output:0*
T0	=
ShuffleDataset/seedConst*
dtype0	*
value	B	 R >
ShuffleDataset/seed2Const*
value	B	 R *
dtype0	ф
ShuffleDatasetShuffleDatasetTensorSliceDataset:handle:0ShuffleDataset/Maximum:z:0ShuffleDataset/seed:output:0ShuffleDataset/seed2:output:0*
output_shapes
: *
reshuffle_each_iteration(*
output_types
2N
#ShuffleAndRepeatDataset/buffer_sizeConst*
dtype0	*
value
B	 R H
ShuffleAndRepeatDataset/seed_1Const*
dtype0	*
value	B	 R I
ShuffleAndRepeatDataset/seed2_1Const*
dtype0	*
value	B	 R G
ShuffleAndRepeatDataset/countConst*
value	B	 R
*
dtype0	Ђ
ShuffleAndRepeatDatasetShuffleAndRepeatDatasetShuffleDataset:handle:0,ShuffleAndRepeatDataset/buffer_size:output:0'ShuffleAndRepeatDataset/seed_1:output:0(ShuffleAndRepeatDataset/seed2_1:output:0&ShuffleAndRepeatDataset/count:output:0*
output_shapes
: *
output_types
2Ћ

MapDataset
MapDataset ShuffleAndRepeatDataset:handle:0*

Targuments
 *#
output_shapes
:: * 
fR
tf_map_func_6yLLF6rv0aU*
output_types
2A
BatchDataset/batch_sizeConst*
dtype0	*
value	B	 R@Њ
BatchDatasetBatchDatasetMapDataset:handle:0 BatchDataset/batch_size:output:0*
output_types
2*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџH
PrefetchDataset/buffer_size_1Const*
dtype0	*
value
B	 R И
PrefetchDatasetPrefetchDatasetBatchDataset:handle:0&PrefetchDataset/buffer_size_1:output:0*
output_types
2*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ"+
prefetchdatasetPrefetchDataset:handle:0"сфЩЇ!г      ўБћ	Y1~#>ЩжAJІ
+ї*
:
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T" 
Ttype:
2	"
use_lockingbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype

IteratorGetNext
iterator

components2output_types"
output_types
list(type)(0" 
output_shapeslist(shape)(0
C
IteratorToStringHandle
resource_handle
string_handle
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
+
MatchingFiles
pattern
	filenames
д
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ю
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
Џ
OneShotIterator

handle"
dataset_factoryfunc"
output_types
list(type)(0" 
output_shapeslist(shape)(0"
	containerstring "
shared_namestring 
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
\
	RefSwitch
data"T
pred

output_false"T
output_true"T"	
Ttype
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	

#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.8.02
b'unknown'т

global_step/Initializer/zerosConst*
dtype0	*
_output_shapes
: *
_class
loc:@global_step*
value	B	 R 

global_step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step

!global_step/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_output_shapes
: *
_class
loc:@global_step

global_step/cond/SwitchSwitch!global_step/IsVariableInitialized!global_step/IsVariableInitialized*
_output_shapes
: : *
T0

a
global_step/cond/switch_tIdentityglobal_step/cond/Switch:1*
T0
*
_output_shapes
: 
_
global_step/cond/switch_fIdentityglobal_step/cond/Switch*
T0
*
_output_shapes
: 
h
global_step/cond/pred_idIdentity!global_step/IsVariableInitialized*
_output_shapes
: *
T0

b
global_step/cond/readIdentityglobal_step/cond/read/Switch:1*
_output_shapes
: *
T0	

global_step/cond/read/Switch	RefSwitchglobal_stepglobal_step/cond/pred_id*
_output_shapes
: : *
T0	*
_class
loc:@global_step

global_step/cond/Switch_1Switchglobal_step/Initializer/zerosglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 
}
global_step/cond/MergeMergeglobal_step/cond/Switch_1global_step/cond/read*
N*
_output_shapes
: : *
T0	
S
global_step/add/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
b
global_step/addAddglobal_step/cond/Mergeglobal_step/add/y*
T0	*
_output_shapes
: 

MatchingFiles/patternConst"/device:CPU:0*
dtype0*
_output_shapes
: *.
value%B# B/data/dogscats/train/**/*.jpg
i
MatchingFilesMatchingFilesMatchingFiles/pattern"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ
a
ShapeShapeMatchingFiles"/device:CPU:0*
T0*
out_type0	*
_output_shapes
:
l
strided_slice/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 
n
strided_slice/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
n
strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0	
Z
	Maximum/yConst"/device:CPU:0*
value	B	 R*
dtype0	*
_output_shapes
: 
\
MaximumMaximumstrided_slice	Maximum/y"/device:CPU:0*
_output_shapes
: *
T0	
U
seedConst"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
V
seed2Const"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
]
buffer_sizeConst"/device:CPU:0*
dtype0	*
_output_shapes
: *
value
B	 R 
V
countConst"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R

W
seed_1Const"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
X
seed2_1Const"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
[

batch_sizeConst"/device:CPU:0*
value	B	 R@*
dtype0	*
_output_shapes
: 
_
buffer_size_1Const"/device:CPU:0*
value
B	 R *
dtype0	*
_output_shapes
: 
і
OneShotIteratorOneShotIterator"/device:CPU:0*
shared_name *=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*
	container *
output_types
2*
_output_shapes
: *0
dataset_factoryR
_make_dataset_GUdGPfB2wf0
h
IteratorToStringHandleIteratorToStringHandleOneShotIterator"/device:CPU:0*
_output_shapes
: 
й
IteratorGetNextIteratorGetNextOneShotIterator"/device:CPU:0*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*>
_output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*
output_types
2
w
layer_1/truncated_normal/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
b
layer_1/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
d
layer_1/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
В
(layer_1/truncated_normal/TruncatedNormalTruncatedNormallayer_1/truncated_normal/shape*
T0*
dtype0*&
_output_shapes
: *
seed2 *

seed 

layer_1/truncated_normal/mulMul(layer_1/truncated_normal/TruncatedNormallayer_1/truncated_normal/stddev*&
_output_shapes
: *
T0

layer_1/truncated_normalAddlayer_1/truncated_normal/mullayer_1/truncated_normal/mean*&
_output_shapes
: *
T0

layer_1/Variable
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
Ь
layer_1/Variable/AssignAssignlayer_1/Variablelayer_1/truncated_normal*
use_locking(*
T0*#
_class
loc:@layer_1/Variable*
validate_shape(*&
_output_shapes
: 

layer_1/Variable/readIdentitylayer_1/Variable*&
_output_shapes
: *
T0*#
_class
loc:@layer_1/Variable
Z
layer_1/ConstConst*
dtype0*
_output_shapes
: *
valueB *ЭЬЬ=
~
layer_1/Variable_1
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Л
layer_1/Variable_1/AssignAssignlayer_1/Variable_1layer_1/Const*
use_locking(*
T0*%
_class
loc:@layer_1/Variable_1*
validate_shape(*
_output_shapes
: 

layer_1/Variable_1/readIdentitylayer_1/Variable_1*
_output_shapes
: *
T0*%
_class
loc:@layer_1/Variable_1
ш
layer_1/Conv2DConv2DIteratorGetNextlayer_1/Variable/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
u
layer_1/addAddlayer_1/Conv2Dlayer_1/Variable_1/read*
T0*/
_output_shapes
:џџџџџџџџџ 
[
layer_1/ReluRelulayer_1/add*/
_output_shapes
:џџџџџџџџџ *
T0
Д
layer_1/MaxPoolMaxPoollayer_1/Relu*/
_output_shapes
:џџџџџџџџџ *
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
w
layer_2/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"          @   
b
layer_2/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
d
layer_2/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=
В
(layer_2/truncated_normal/TruncatedNormalTruncatedNormallayer_2/truncated_normal/shape*
dtype0*&
_output_shapes
: @*
seed2 *

seed *
T0

layer_2/truncated_normal/mulMul(layer_2/truncated_normal/TruncatedNormallayer_2/truncated_normal/stddev*&
_output_shapes
: @*
T0

layer_2/truncated_normalAddlayer_2/truncated_normal/mullayer_2/truncated_normal/mean*&
_output_shapes
: @*
T0

layer_2/Variable
VariableV2*
shared_name *
dtype0*&
_output_shapes
: @*
	container *
shape: @
Ь
layer_2/Variable/AssignAssignlayer_2/Variablelayer_2/truncated_normal*
use_locking(*
T0*#
_class
loc:@layer_2/Variable*
validate_shape(*&
_output_shapes
: @

layer_2/Variable/readIdentitylayer_2/Variable*&
_output_shapes
: @*
T0*#
_class
loc:@layer_2/Variable
Z
layer_2/ConstConst*
valueB@*ЭЬЬ=*
dtype0*
_output_shapes
:@
~
layer_2/Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes
:@*
	container *
shape:@
Л
layer_2/Variable_1/AssignAssignlayer_2/Variable_1layer_2/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@layer_2/Variable_1

layer_2/Variable_1/readIdentitylayer_2/Variable_1*
T0*%
_class
loc:@layer_2/Variable_1*
_output_shapes
:@
ш
layer_2/Conv2DConv2Dlayer_1/MaxPoollayer_2/Variable/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
u
layer_2/addAddlayer_2/Conv2Dlayer_2/Variable_1/read*/
_output_shapes
:џџџџџџџџџ@*
T0
[
layer_2/ReluRelulayer_2/add*/
_output_shapes
:џџџџџџџџџ@*
T0
Д
layer_2/MaxPoolMaxPoollayer_2/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
T0
q
 fc_layer1/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"@     
d
fc_layer1/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
f
!fc_layer1/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=
А
*fc_layer1/truncated_normal/TruncatedNormalTruncatedNormal fc_layer1/truncated_normal/shape*
dtype0* 
_output_shapes
:
Р*
seed2 *

seed *
T0

fc_layer1/truncated_normal/mulMul*fc_layer1/truncated_normal/TruncatedNormal!fc_layer1/truncated_normal/stddev* 
_output_shapes
:
Р*
T0

fc_layer1/truncated_normalAddfc_layer1/truncated_normal/mulfc_layer1/truncated_normal/mean* 
_output_shapes
:
Р*
T0

fc_layer1/Variable
VariableV2*
dtype0* 
_output_shapes
:
Р*
	container *
shape:
Р*
shared_name 
Ю
fc_layer1/Variable/AssignAssignfc_layer1/Variablefc_layer1/truncated_normal*
validate_shape(* 
_output_shapes
:
Р*
use_locking(*
T0*%
_class
loc:@fc_layer1/Variable

fc_layer1/Variable/readIdentityfc_layer1/Variable* 
_output_shapes
:
Р*
T0*%
_class
loc:@fc_layer1/Variable
^
fc_layer1/ConstConst*
valueB*ЭЬЬ=*
dtype0*
_output_shapes	
:

fc_layer1/Variable_1
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ф
fc_layer1/Variable_1/AssignAssignfc_layer1/Variable_1fc_layer1/Const*
use_locking(*
T0*'
_class
loc:@fc_layer1/Variable_1*
validate_shape(*
_output_shapes	
:

fc_layer1/Variable_1/readIdentityfc_layer1/Variable_1*
_output_shapes	
:*
T0*'
_class
loc:@fc_layer1/Variable_1
h
fc_layer1/Reshape/shapeConst*
valueB"џџџџ@  *
dtype0*
_output_shapes
:

fc_layer1/ReshapeReshapelayer_2/MaxPoolfc_layer1/Reshape/shape*(
_output_shapes
:џџџџџџџџџР*
T0*
Tshape0

fc_layer1/MatMulMatMulfc_layer1/Reshapefc_layer1/Variable/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
t
fc_layer1/addAddfc_layer1/MatMulfc_layer1/Variable_1/read*(
_output_shapes
:џџџџџџџџџ*
T0
X
fc_layer1/ReluRelufc_layer1/add*(
_output_shapes
:џџџџџџџџџ*
T0
t
#read_out_fc2/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
g
"read_out_fc2/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$read_out_fc2/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=
Е
-read_out_fc2/truncated_normal/TruncatedNormalTruncatedNormal#read_out_fc2/truncated_normal/shape*
T0*
dtype0*
_output_shapes
:	
*
seed2 *

seed 
Ї
!read_out_fc2/truncated_normal/mulMul-read_out_fc2/truncated_normal/TruncatedNormal$read_out_fc2/truncated_normal/stddev*
T0*
_output_shapes
:	


read_out_fc2/truncated_normalAdd!read_out_fc2/truncated_normal/mul"read_out_fc2/truncated_normal/mean*
_output_shapes
:	
*
T0

read_out_fc2/Variable
VariableV2*
dtype0*
_output_shapes
:	
*
	container *
shape:	
*
shared_name 
й
read_out_fc2/Variable/AssignAssignread_out_fc2/Variableread_out_fc2/truncated_normal*
use_locking(*
T0*(
_class
loc:@read_out_fc2/Variable*
validate_shape(*
_output_shapes
:	


read_out_fc2/Variable/readIdentityread_out_fc2/Variable*
_output_shapes
:	
*
T0*(
_class
loc:@read_out_fc2/Variable
_
read_out_fc2/ConstConst*
valueB
*ЭЬЬ=*
dtype0*
_output_shapes
:


read_out_fc2/Variable_1
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 
Я
read_out_fc2/Variable_1/AssignAssignread_out_fc2/Variable_1read_out_fc2/Const*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0**
_class 
loc:@read_out_fc2/Variable_1

read_out_fc2/Variable_1/readIdentityread_out_fc2/Variable_1*
_output_shapes
:
*
T0**
_class 
loc:@read_out_fc2/Variable_1
Ё
read_out_fc2/MatMulMatMulfc_layer1/Reluread_out_fc2/Variable/read*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
transpose_b( *
T0
|
read_out_fc2/addAddread_out_fc2/MatMulread_out_fc2/Variable_1/read*
T0*'
_output_shapes
:џџџџџџџџџ

z
)SparseSoftmaxCrossEntropyWithLogits/ShapeShapeIteratorGetNext:1*
T0*
out_type0*
_output_shapes
:
у
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsread_out_fc2/addIteratorGetNext:1*
T0*6
_output_shapes$
":џџџџџџџџџ:џџџџџџџџџ
*
Tlabels0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:

MeanMeanGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :

ArgMaxArgMaxread_out_fc2/addArgMax/dimension*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
Q
CastCastArgMax*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
U
EqualEqualCastIteratorGetNext:1*
T0*#
_output_shapes
:џџџџџџџџџ
S
ToFloatCastEqual*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0


acc_op/total/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class
loc:@acc_op/total*
valueB
 *    

acc_op/total
VariableV2*
shared_name *
_class
loc:@acc_op/total*
	container *
shape: *
dtype0*
_output_shapes
: 
Ж
acc_op/total/AssignAssignacc_op/totalacc_op/total/Initializer/zeros*
use_locking(*
T0*
_class
loc:@acc_op/total*
validate_shape(*
_output_shapes
: 
m
acc_op/total/readIdentityacc_op/total*
T0*
_class
loc:@acc_op/total*
_output_shapes
: 

acc_op/count/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class
loc:@acc_op/count*
valueB
 *    

acc_op/count
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@acc_op/count*
	container *
shape: 
Ж
acc_op/count/AssignAssignacc_op/countacc_op/count/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@acc_op/count
m
acc_op/count/readIdentityacc_op/count*
_output_shapes
: *
T0*
_class
loc:@acc_op/count
M
acc_op/SizeSizeToFloat*
_output_shapes
: *
T0*
out_type0
U
acc_op/ToFloat_1Castacc_op/Size*

SrcT0*
_output_shapes
: *

DstT0
V
acc_op/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
f

acc_op/SumSumToFloatacc_op/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

acc_op/AssignAdd	AssignAddacc_op/total
acc_op/Sum*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@acc_op/total

acc_op/AssignAdd_1	AssignAddacc_op/countacc_op/ToFloat_1^ToFloat*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@acc_op/count
`
acc_op/truedivRealDivacc_op/total/readacc_op/count/read*
T0*
_output_shapes
: 
V
acc_op/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
`
acc_op/GreaterGreateracc_op/count/readacc_op/zeros_like*
T0*
_output_shapes
: 
j
acc_op/valueSelectacc_op/Greateracc_op/truedivacc_op/zeros_like*
T0*
_output_shapes
: 
b
acc_op/truediv_1RealDivacc_op/AssignAddacc_op/AssignAdd_1*
_output_shapes
: *
T0
X
acc_op/zeros_like_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
e
acc_op/Greater_1Greateracc_op/AssignAdd_1acc_op/zeros_like_1*
T0*
_output_shapes
: 
t
acc_op/update_opSelectacc_op/Greater_1acc_op/truediv_1acc_op/zeros_like_1*
T0*
_output_shapes
: 
V
accuracy/tagsConst*
dtype0*
_output_shapes
: *
valueB Baccuracy
[
accuracyScalarSummaryaccuracy/tagsacc_op/update_op*
_output_shapes
: *
T0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
 
gradients/Mean_grad/ShapeShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ
Ђ
gradients/Mean_grad/Shape_1ShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ

gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:џџџџџџџџџ
*
T0
­
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:џџџџџџџџџ
*Д
messageЈЅCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
А
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
Б
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Mean_grad/truedivegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
о
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:џџџџџџџџџ

x
%gradients/read_out_fc2/add_grad/ShapeShaperead_out_fc2/MatMul*
_output_shapes
:*
T0*
out_type0
q
'gradients/read_out_fc2/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
л
5gradients/read_out_fc2/add_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/read_out_fc2/add_grad/Shape'gradients/read_out_fc2/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
§
#gradients/read_out_fc2/add_grad/SumSumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul5gradients/read_out_fc2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
О
'gradients/read_out_fc2/add_grad/ReshapeReshape#gradients/read_out_fc2/add_grad/Sum%gradients/read_out_fc2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ


%gradients/read_out_fc2/add_grad/Sum_1SumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul7gradients/read_out_fc2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
)gradients/read_out_fc2/add_grad/Reshape_1Reshape%gradients/read_out_fc2/add_grad/Sum_1'gradients/read_out_fc2/add_grad/Shape_1*
_output_shapes
:
*
T0*
Tshape0

0gradients/read_out_fc2/add_grad/tuple/group_depsNoOp(^gradients/read_out_fc2/add_grad/Reshape*^gradients/read_out_fc2/add_grad/Reshape_1

8gradients/read_out_fc2/add_grad/tuple/control_dependencyIdentity'gradients/read_out_fc2/add_grad/Reshape1^gradients/read_out_fc2/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/read_out_fc2/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ


:gradients/read_out_fc2/add_grad/tuple/control_dependency_1Identity)gradients/read_out_fc2/add_grad/Reshape_11^gradients/read_out_fc2/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/read_out_fc2/add_grad/Reshape_1*
_output_shapes
:

т
)gradients/read_out_fc2/MatMul_grad/MatMulMatMul8gradients/read_out_fc2/add_grad/tuple/control_dependencyread_out_fc2/Variable/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
Я
+gradients/read_out_fc2/MatMul_grad/MatMul_1MatMulfc_layer1/Relu8gradients/read_out_fc2/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	
*
transpose_a(*
transpose_b( 

3gradients/read_out_fc2/MatMul_grad/tuple/group_depsNoOp*^gradients/read_out_fc2/MatMul_grad/MatMul,^gradients/read_out_fc2/MatMul_grad/MatMul_1

;gradients/read_out_fc2/MatMul_grad/tuple/control_dependencyIdentity)gradients/read_out_fc2/MatMul_grad/MatMul4^gradients/read_out_fc2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/read_out_fc2/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџ

=gradients/read_out_fc2/MatMul_grad/tuple/control_dependency_1Identity+gradients/read_out_fc2/MatMul_grad/MatMul_14^gradients/read_out_fc2/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/read_out_fc2/MatMul_grad/MatMul_1*
_output_shapes
:	

В
&gradients/fc_layer1/Relu_grad/ReluGradReluGrad;gradients/read_out_fc2/MatMul_grad/tuple/control_dependencyfc_layer1/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
r
"gradients/fc_layer1/add_grad/ShapeShapefc_layer1/MatMul*
_output_shapes
:*
T0*
out_type0
o
$gradients/fc_layer1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
в
2gradients/fc_layer1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/fc_layer1/add_grad/Shape$gradients/fc_layer1/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
У
 gradients/fc_layer1/add_grad/SumSum&gradients/fc_layer1/Relu_grad/ReluGrad2gradients/fc_layer1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
$gradients/fc_layer1/add_grad/ReshapeReshape gradients/fc_layer1/add_grad/Sum"gradients/fc_layer1/add_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ч
"gradients/fc_layer1/add_grad/Sum_1Sum&gradients/fc_layer1/Relu_grad/ReluGrad4gradients/fc_layer1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Џ
&gradients/fc_layer1/add_grad/Reshape_1Reshape"gradients/fc_layer1/add_grad/Sum_1$gradients/fc_layer1/add_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

-gradients/fc_layer1/add_grad/tuple/group_depsNoOp%^gradients/fc_layer1/add_grad/Reshape'^gradients/fc_layer1/add_grad/Reshape_1

5gradients/fc_layer1/add_grad/tuple/control_dependencyIdentity$gradients/fc_layer1/add_grad/Reshape.^gradients/fc_layer1/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/fc_layer1/add_grad/Reshape*(
_output_shapes
:џџџџџџџџџ
ќ
7gradients/fc_layer1/add_grad/tuple/control_dependency_1Identity&gradients/fc_layer1/add_grad/Reshape_1.^gradients/fc_layer1/add_grad/tuple/group_deps*
_output_shapes	
:*
T0*9
_class/
-+loc:@gradients/fc_layer1/add_grad/Reshape_1
й
&gradients/fc_layer1/MatMul_grad/MatMulMatMul5gradients/fc_layer1/add_grad/tuple/control_dependencyfc_layer1/Variable/read*(
_output_shapes
:џџџџџџџџџР*
transpose_a( *
transpose_b(*
T0
Э
(gradients/fc_layer1/MatMul_grad/MatMul_1MatMulfc_layer1/Reshape5gradients/fc_layer1/add_grad/tuple/control_dependency* 
_output_shapes
:
Р*
transpose_a(*
transpose_b( *
T0

0gradients/fc_layer1/MatMul_grad/tuple/group_depsNoOp'^gradients/fc_layer1/MatMul_grad/MatMul)^gradients/fc_layer1/MatMul_grad/MatMul_1

8gradients/fc_layer1/MatMul_grad/tuple/control_dependencyIdentity&gradients/fc_layer1/MatMul_grad/MatMul1^gradients/fc_layer1/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/fc_layer1/MatMul_grad/MatMul*(
_output_shapes
:џџџџџџџџџР

:gradients/fc_layer1/MatMul_grad/tuple/control_dependency_1Identity(gradients/fc_layer1/MatMul_grad/MatMul_11^gradients/fc_layer1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/fc_layer1/MatMul_grad/MatMul_1* 
_output_shapes
:
Р
u
&gradients/fc_layer1/Reshape_grad/ShapeShapelayer_2/MaxPool*
_output_shapes
:*
T0*
out_type0
н
(gradients/fc_layer1/Reshape_grad/ReshapeReshape8gradients/fc_layer1/MatMul_grad/tuple/control_dependency&gradients/fc_layer1/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ@

*gradients/layer_2/MaxPool_grad/MaxPoolGradMaxPoolGradlayer_2/Relulayer_2/MaxPool(gradients/fc_layer1/Reshape_grad/Reshape*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@
Є
$gradients/layer_2/Relu_grad/ReluGradReluGrad*gradients/layer_2/MaxPool_grad/MaxPoolGradlayer_2/Relu*/
_output_shapes
:џџџџџџџџџ@*
T0
n
 gradients/layer_2/add_grad/ShapeShapelayer_2/Conv2D*
_output_shapes
:*
T0*
out_type0
l
"gradients/layer_2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@
Ь
0gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/layer_2/add_grad/Shape"gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
gradients/layer_2/add_grad/SumSum$gradients/layer_2/Relu_grad/ReluGrad0gradients/layer_2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
"gradients/layer_2/add_grad/ReshapeReshapegradients/layer_2/add_grad/Sum gradients/layer_2/add_grad/Shape*/
_output_shapes
:џџџџџџџџџ@*
T0*
Tshape0
С
 gradients/layer_2/add_grad/Sum_1Sum$gradients/layer_2/Relu_grad/ReluGrad2gradients/layer_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ј
$gradients/layer_2/add_grad/Reshape_1Reshape gradients/layer_2/add_grad/Sum_1"gradients/layer_2/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:@

+gradients/layer_2/add_grad/tuple/group_depsNoOp#^gradients/layer_2/add_grad/Reshape%^gradients/layer_2/add_grad/Reshape_1

3gradients/layer_2/add_grad/tuple/control_dependencyIdentity"gradients/layer_2/add_grad/Reshape,^gradients/layer_2/add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/layer_2/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџ@
ѓ
5gradients/layer_2/add_grad/tuple/control_dependency_1Identity$gradients/layer_2/add_grad/Reshape_1,^gradients/layer_2/add_grad/tuple/group_deps*
_output_shapes
:@*
T0*7
_class-
+)loc:@gradients/layer_2/add_grad/Reshape_1

$gradients/layer_2/Conv2D_grad/ShapeNShapeNlayer_1/MaxPoollayer_2/Variable/read*
T0*
out_type0*
N* 
_output_shapes
::
|
#gradients/layer_2/Conv2D_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"          @   
§
1gradients/layer_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/layer_2/Conv2D_grad/ShapeNlayer_2/Variable/read3gradients/layer_2/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д
2gradients/layer_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterlayer_1/MaxPool#gradients/layer_2/Conv2D_grad/Const3gradients/layer_2/add_grad/tuple/control_dependency*&
_output_shapes
: @*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME

.gradients/layer_2/Conv2D_grad/tuple/group_depsNoOp3^gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter2^gradients/layer_2/Conv2D_grad/Conv2DBackpropInput
І
6gradients/layer_2/Conv2D_grad/tuple/control_dependencyIdentity1gradients/layer_2/Conv2D_grad/Conv2DBackpropInput/^gradients/layer_2/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/layer_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ 
Ё
8gradients/layer_2/Conv2D_grad/tuple/control_dependency_1Identity2gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter/^gradients/layer_2/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @

*gradients/layer_1/MaxPool_grad/MaxPoolGradMaxPoolGradlayer_1/Relulayer_1/MaxPool6gradients/layer_2/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
T0
Є
$gradients/layer_1/Relu_grad/ReluGradReluGrad*gradients/layer_1/MaxPool_grad/MaxPoolGradlayer_1/Relu*
T0*/
_output_shapes
:џџџџџџџџџ 
n
 gradients/layer_1/add_grad/ShapeShapelayer_1/Conv2D*
T0*
out_type0*
_output_shapes
:
l
"gradients/layer_1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB: 
Ь
0gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/layer_1/add_grad/Shape"gradients/layer_1/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Н
gradients/layer_1/add_grad/SumSum$gradients/layer_1/Relu_grad/ReluGrad0gradients/layer_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
"gradients/layer_1/add_grad/ReshapeReshapegradients/layer_1/add_grad/Sum gradients/layer_1/add_grad/Shape*/
_output_shapes
:џџџџџџџџџ *
T0*
Tshape0
С
 gradients/layer_1/add_grad/Sum_1Sum$gradients/layer_1/Relu_grad/ReluGrad2gradients/layer_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ј
$gradients/layer_1/add_grad/Reshape_1Reshape gradients/layer_1/add_grad/Sum_1"gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

+gradients/layer_1/add_grad/tuple/group_depsNoOp#^gradients/layer_1/add_grad/Reshape%^gradients/layer_1/add_grad/Reshape_1

3gradients/layer_1/add_grad/tuple/control_dependencyIdentity"gradients/layer_1/add_grad/Reshape,^gradients/layer_1/add_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ *
T0*5
_class+
)'loc:@gradients/layer_1/add_grad/Reshape
ѓ
5gradients/layer_1/add_grad/tuple/control_dependency_1Identity$gradients/layer_1/add_grad/Reshape_1,^gradients/layer_1/add_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/layer_1/add_grad/Reshape_1

$gradients/layer_1/Conv2D_grad/ShapeNShapeNIteratorGetNextlayer_1/Variable/read*
N* 
_output_shapes
::*
T0*
out_type0
|
#gradients/layer_1/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
§
1gradients/layer_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/layer_1/Conv2D_grad/ShapeNlayer_1/Variable/read3gradients/layer_1/add_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
д
2gradients/layer_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterIteratorGetNext#gradients/layer_1/Conv2D_grad/Const3gradients/layer_1/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 

.gradients/layer_1/Conv2D_grad/tuple/group_depsNoOp3^gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter2^gradients/layer_1/Conv2D_grad/Conv2DBackpropInput
І
6gradients/layer_1/Conv2D_grad/tuple/control_dependencyIdentity1gradients/layer_1/Conv2D_grad/Conv2DBackpropInput/^gradients/layer_1/Conv2D_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*D
_class:
86loc:@gradients/layer_1/Conv2D_grad/Conv2DBackpropInput
Ё
8gradients/layer_1/Conv2D_grad/tuple/control_dependency_1Identity2gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter/^gradients/layer_1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
b
GradientDescent/learning_rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ј
<GradientDescent/update_layer_1/Variable/ApplyGradientDescentApplyGradientDescentlayer_1/VariableGradientDescent/learning_rate8gradients/layer_1/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
: *
use_locking( *
T0*#
_class
loc:@layer_1/Variable

>GradientDescent/update_layer_1/Variable_1/ApplyGradientDescentApplyGradientDescentlayer_1/Variable_1GradientDescent/learning_rate5gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@layer_1/Variable_1*
_output_shapes
: 
Ј
<GradientDescent/update_layer_2/Variable/ApplyGradientDescentApplyGradientDescentlayer_2/VariableGradientDescent/learning_rate8gradients/layer_2/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
: @*
use_locking( *
T0*#
_class
loc:@layer_2/Variable

>GradientDescent/update_layer_2/Variable_1/ApplyGradientDescentApplyGradientDescentlayer_2/Variable_1GradientDescent/learning_rate5gradients/layer_2/add_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*%
_class
loc:@layer_2/Variable_1
Њ
>GradientDescent/update_fc_layer1/Variable/ApplyGradientDescentApplyGradientDescentfc_layer1/VariableGradientDescent/learning_rate:gradients/fc_layer1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@fc_layer1/Variable* 
_output_shapes
:
Р
Ј
@GradientDescent/update_fc_layer1/Variable_1/ApplyGradientDescentApplyGradientDescentfc_layer1/Variable_1GradientDescent/learning_rate7gradients/fc_layer1/add_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*'
_class
loc:@fc_layer1/Variable_1
Е
AGradientDescent/update_read_out_fc2/Variable/ApplyGradientDescentApplyGradientDescentread_out_fc2/VariableGradientDescent/learning_rate=gradients/read_out_fc2/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	
*
use_locking( *
T0*(
_class
loc:@read_out_fc2/Variable
Г
CGradientDescent/update_read_out_fc2/Variable_1/ApplyGradientDescentApplyGradientDescentread_out_fc2/Variable_1GradientDescent/learning_rate:gradients/read_out_fc2/add_grad/tuple/control_dependency_1*
_output_shapes
:
*
use_locking( *
T0**
_class 
loc:@read_out_fc2/Variable_1
Ќ
GradientDescent/updateNoOp?^GradientDescent/update_fc_layer1/Variable/ApplyGradientDescentA^GradientDescent/update_fc_layer1/Variable_1/ApplyGradientDescent=^GradientDescent/update_layer_1/Variable/ApplyGradientDescent?^GradientDescent/update_layer_1/Variable_1/ApplyGradientDescent=^GradientDescent/update_layer_2/Variable/ApplyGradientDescent?^GradientDescent/update_layer_2/Variable_1/ApplyGradientDescentB^GradientDescent/update_read_out_fc2/Variable/ApplyGradientDescentD^GradientDescent/update_read_out_fc2/Variable_1/ApplyGradientDescent

GradientDescent/valueConst^GradientDescent/update*
_class
loc:@global_step*
value	B	 R*
dtype0	*
_output_shapes
: 

GradientDescent	AssignAddglobal_stepGradientDescent/value*
_output_shapes
: *
use_locking( *
T0	*
_class
loc:@global_step
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
_output_shapes
: *
T0У+
ю

tf_map_func_6yLLF6rv0aU
arg0
resize_images_squeeze

cond_merge25A wrapper for Defun that facilitates shape inference./
ConstConst*
value	B B/*
dtype02
packedPackarg0*
N*
T0*

axis M
StringSplitStringSplitpacked:output:0Const:output:0*

skip_empty(J
strided_slice/stackConst*
valueB:
ўџџџџџџџџ*
dtype0L
strided_slice/stack_1Const*
valueB:
џџџџџџџџџ*
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:
strided_sliceStridedSliceStringSplit:values:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 4
Equal/yConst*
dtype0*
valueB
 BdogsA
EqualEqualstrided_slice:output:0Equal/y:output:0*
T04
cond/SwitchSwitch	Equal:z:0	Equal:z:0*
T0
=
cond/switch_tIdentitycond/Switch:output_true:0*
T0
>
cond/switch_fIdentitycond/Switch:output_false:0*
T0
,
cond/pred_idIdentity	Equal:z:0*
T0
D

cond/ConstConst^cond/switch_t*
dtype0*
value	B : F
cond/Const_1Const^cond/switch_f*
dtype0*
value	B :Q

cond/MergeMergecond/Const_1:output:0cond/Const:output:0*
T0*
N
ReadFileReadFilearg0Ў

DecodeJpeg
DecodeJpegReadFile:contents:0*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( F
convert_image/CastCastDecodeJpeg:image:0*

SrcT0*

DstT0<
convert_image/yConst*
valueB
 *;*
dtype0O
convert_imageMulconvert_image/Cast:y:0convert_image/y:output:0*
T0F
resize_images/ExpandDims/dimConst*
value	B : *
dtype0u
resize_images/ExpandDims
ExpandDimsconvert_image:z:0%resize_images/ExpandDims/dim:output:0*

Tdim0*
T0G
resize_images/sizeConst*
valueB"      *
dtype0
resize_images/ResizeBilinearResizeBilinear!resize_images/ExpandDims:output:0resize_images/size:output:0*
align_corners( *
T0o
resize_images/SqueezeSqueeze-resize_images/ResizeBilinear:resized_images:0*
squeeze_dims
 *
T0"!

cond_mergecond/Merge:output:0"7
resize_images_squeezeresize_images/Squeeze:output:0
Я
3
_make_dataset_GUdGPfB2wf0
prefetchdatasetn
(TensorSliceDataset/MatchingFiles/patternConst*
dtype0*.
value%B# B/data/dogscats/train/**/*.jpgd
 TensorSliceDataset/MatchingFilesMatchingFiles1TensorSliceDataset/MatchingFiles/pattern:output:0
TensorSliceDatasetTensorSliceDataset,TensorSliceDataset/MatchingFiles:filenames:0*
output_shapes
: *
Toutput_types
2j
$ShuffleDataset/MatchingFiles/patternConst*
dtype0*.
value%B# B/data/dogscats/train/**/*.jpg\
ShuffleDataset/MatchingFilesMatchingFiles-ShuffleDataset/MatchingFiles/pattern:output:0`
ShuffleDataset/ShapeShape(ShuffleDataset/MatchingFiles:filenames:0*
T0*
out_type0	P
"ShuffleDataset/strided_slice/stackConst*
dtype0*
valueB: R
$ShuffleDataset/strided_slice/stack_1Const*
dtype0*
valueB:R
$ShuffleDataset/strided_slice/stack_2Const*
valueB:*
dtype0а
ShuffleDataset/strided_sliceStridedSliceShuffleDataset/Shape:output:0+ShuffleDataset/strided_slice/stack:output:0-ShuffleDataset/strided_slice/stack_1:output:0-ShuffleDataset/strided_slice/stack_2:output:0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0	B
ShuffleDataset/Maximum/yConst*
dtype0	*
value	B	 Rt
ShuffleDataset/MaximumMaximum%ShuffleDataset/strided_slice:output:0!ShuffleDataset/Maximum/y:output:0*
T0	=
ShuffleDataset/seedConst*
value	B	 R *
dtype0	>
ShuffleDataset/seed2Const*
value	B	 R *
dtype0	ф
ShuffleDatasetShuffleDatasetTensorSliceDataset:handle:0ShuffleDataset/Maximum:z:0ShuffleDataset/seed:output:0ShuffleDataset/seed2:output:0*
output_shapes
: *
reshuffle_each_iteration(*
output_types
2N
#ShuffleAndRepeatDataset/buffer_sizeConst*
dtype0	*
value
B	 R H
ShuffleAndRepeatDataset/seed_1Const*
dtype0	*
value	B	 R I
ShuffleAndRepeatDataset/seed2_1Const*
value	B	 R *
dtype0	G
ShuffleAndRepeatDataset/countConst*
value	B	 R
*
dtype0	Ђ
ShuffleAndRepeatDatasetShuffleAndRepeatDatasetShuffleDataset:handle:0,ShuffleAndRepeatDataset/buffer_size:output:0'ShuffleAndRepeatDataset/seed_1:output:0(ShuffleAndRepeatDataset/seed2_1:output:0&ShuffleAndRepeatDataset/count:output:0*
output_shapes
: *
output_types
2Ћ

MapDataset
MapDataset ShuffleAndRepeatDataset:handle:0* 
fR
tf_map_func_6yLLF6rv0aU*
output_types
2*

Targuments
 *#
output_shapes
:: A
BatchDataset/batch_sizeConst*
dtype0	*
value	B	 R@Њ
BatchDatasetBatchDatasetMapDataset:handle:0 BatchDataset/batch_size:output:0*
output_types
2*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџH
PrefetchDataset/buffer_size_1Const*
value
B	 R *
dtype0	И
PrefetchDatasetPrefetchDatasetBatchDataset:handle:0&PrefetchDataset/buffer_size_1:output:0*
output_types
2*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ"+
prefetchdatasetPrefetchDataset:handle:0""

savers "
losses


Mean:0"
train_op

GradientDescent"6
metric_variables"
 
acc_op/total:0
acc_op/count:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"Р
cond_contextЏЌ

global_step/cond/cond_textglobal_step/cond/pred_id:0global_step/cond/switch_t:0 *Ј
global_step/cond/pred_id:0
global_step/cond/read/Switch:1
global_step/cond/read:0
global_step/cond/switch_t:0
global_step:0/
global_step:0global_step/cond/read/Switch:18
global_step/cond/pred_id:0global_step/cond/pred_id:0:
global_step/cond/switch_t:0global_step/cond/switch_t:0
Є
global_step/cond/cond_text_1global_step/cond/pred_id:0global_step/cond/switch_f:0*Ъ
global_step/Initializer/zeros:0
global_step/cond/Switch_1:0
global_step/cond/Switch_1:1
global_step/cond/pred_id:0
global_step/cond/switch_f:08
global_step/cond/pred_id:0global_step/cond/pred_id:0>
global_step/Initializer/zeros:0global_step/cond/Switch_1:0:
global_step/cond/switch_f:0global_step/cond/switch_f:0"2
global_step_read_op_cache

global_step/add:0"#
	summaries


accuracy:0
loss:0"п
trainable_variablesЧФ
b
layer_1/Variable:0layer_1/Variable/Assignlayer_1/Variable/read:02layer_1/truncated_normal:0
]
layer_1/Variable_1:0layer_1/Variable_1/Assignlayer_1/Variable_1/read:02layer_1/Const:0
b
layer_2/Variable:0layer_2/Variable/Assignlayer_2/Variable/read:02layer_2/truncated_normal:0
]
layer_2/Variable_1:0layer_2/Variable_1/Assignlayer_2/Variable_1/read:02layer_2/Const:0
j
fc_layer1/Variable:0fc_layer1/Variable/Assignfc_layer1/Variable/read:02fc_layer1/truncated_normal:0
e
fc_layer1/Variable_1:0fc_layer1/Variable_1/Assignfc_layer1/Variable_1/read:02fc_layer1/Const:0
v
read_out_fc2/Variable:0read_out_fc2/Variable/Assignread_out_fc2/Variable/read:02read_out_fc2/truncated_normal:0
q
read_out_fc2/Variable_1:0read_out_fc2/Variable_1/Assignread_out_fc2/Variable_1/read:02read_out_fc2/Const:0"г
local_variablesПМ
\
acc_op/total:0acc_op/total/Assignacc_op/total/read:02 acc_op/total/Initializer/zeros:0
\
acc_op/count:0acc_op/count/Assignacc_op/count/read:02 acc_op/count/Initializer/zeros:0"Џ
	variablesЁ
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
b
layer_1/Variable:0layer_1/Variable/Assignlayer_1/Variable/read:02layer_1/truncated_normal:0
]
layer_1/Variable_1:0layer_1/Variable_1/Assignlayer_1/Variable_1/read:02layer_1/Const:0
b
layer_2/Variable:0layer_2/Variable/Assignlayer_2/Variable/read:02layer_2/truncated_normal:0
]
layer_2/Variable_1:0layer_2/Variable_1/Assignlayer_2/Variable_1/read:02layer_2/Const:0
j
fc_layer1/Variable:0fc_layer1/Variable/Assignfc_layer1/Variable/read:02fc_layer1/truncated_normal:0
e
fc_layer1/Variable_1:0fc_layer1/Variable_1/Assignfc_layer1/Variable_1/read:02fc_layer1/Const:0
v
read_out_fc2/Variable:0read_out_fc2/Variable/Assignread_out_fc2/Variable/read:02read_out_fc2/truncated_normal:0
q
read_out_fc2/Variable_1:0read_out_fc2/Variable_1/Assignread_out_fc2/Variable_1/read:02read_out_fc2/Const:0[ W G     №Лб 	%§Є#>ЩжA"К

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 

global_step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
_output_shapes
: *
T0	*
_class
loc:@global_step

!global_step/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_output_shapes
: *
_class
loc:@global_step

global_step/cond/SwitchSwitch!global_step/IsVariableInitialized!global_step/IsVariableInitialized*
_output_shapes
: : *
T0

a
global_step/cond/switch_tIdentityglobal_step/cond/Switch:1*
_output_shapes
: *
T0

_
global_step/cond/switch_fIdentityglobal_step/cond/Switch*
T0
*
_output_shapes
: 
h
global_step/cond/pred_idIdentity!global_step/IsVariableInitialized*
T0
*
_output_shapes
: 
b
global_step/cond/readIdentityglobal_step/cond/read/Switch:1*
T0	*
_output_shapes
: 

global_step/cond/read/Switch	RefSwitchglobal_stepglobal_step/cond/pred_id*
T0	*
_class
loc:@global_step*
_output_shapes
: : 

global_step/cond/Switch_1Switchglobal_step/Initializer/zerosglobal_step/cond/pred_id*
_output_shapes
: : *
T0	*
_class
loc:@global_step
}
global_step/cond/MergeMergeglobal_step/cond/Switch_1global_step/cond/read*
N*
_output_shapes
: : *
T0	
S
global_step/add/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
b
global_step/addAddglobal_step/cond/Mergeglobal_step/add/y*
T0	*
_output_shapes
: 

MatchingFiles/patternConst"/device:CPU:0*
dtype0*
_output_shapes
: *.
value%B# B/data/dogscats/train/**/*.jpg
i
MatchingFilesMatchingFilesMatchingFiles/pattern"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ
a
ShapeShapeMatchingFiles"/device:CPU:0*
T0*
out_type0	*
_output_shapes
:
l
strided_slice/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 
n
strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
n
strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2"/device:CPU:0*
T0	*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Z
	Maximum/yConst"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R
\
MaximumMaximumstrided_slice	Maximum/y"/device:CPU:0*
T0	*
_output_shapes
: 
U
seedConst"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R 
V
seed2Const"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R 
]
buffer_sizeConst"/device:CPU:0*
dtype0	*
_output_shapes
: *
value
B	 R 
V
countConst"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R

W
seed_1Const"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R 
X
seed2_1Const"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
[

batch_sizeConst"/device:CPU:0*
value	B	 R@*
dtype0	*
_output_shapes
: 
_
buffer_size_1Const"/device:CPU:0*
value
B	 R *
dtype0	*
_output_shapes
: 
і
OneShotIteratorOneShotIterator"/device:CPU:0*
_output_shapes
: *0
dataset_factoryR
_make_dataset_GUdGPfB2wf0*
shared_name *=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*
	container *
output_types
2
h
IteratorToStringHandleIteratorToStringHandleOneShotIterator"/device:CPU:0*
_output_shapes
: 
й
IteratorGetNextIteratorGetNextOneShotIterator"/device:CPU:0*>
_output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*
output_types
2*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ
w
layer_1/truncated_normal/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:
b
layer_1/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
layer_1/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *ЭЬЬ=
В
(layer_1/truncated_normal/TruncatedNormalTruncatedNormallayer_1/truncated_normal/shape*
dtype0*&
_output_shapes
: *
seed2 *

seed *
T0

layer_1/truncated_normal/mulMul(layer_1/truncated_normal/TruncatedNormallayer_1/truncated_normal/stddev*
T0*&
_output_shapes
: 

layer_1/truncated_normalAddlayer_1/truncated_normal/mullayer_1/truncated_normal/mean*
T0*&
_output_shapes
: 

layer_1/Variable
VariableV2*
shared_name *
dtype0*&
_output_shapes
: *
	container *
shape: 
Ь
layer_1/Variable/AssignAssignlayer_1/Variablelayer_1/truncated_normal*
use_locking(*
T0*#
_class
loc:@layer_1/Variable*
validate_shape(*&
_output_shapes
: 

layer_1/Variable/readIdentitylayer_1/Variable*&
_output_shapes
: *
T0*#
_class
loc:@layer_1/Variable
Z
layer_1/ConstConst*
valueB *ЭЬЬ=*
dtype0*
_output_shapes
: 
~
layer_1/Variable_1
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Л
layer_1/Variable_1/AssignAssignlayer_1/Variable_1layer_1/Const*
use_locking(*
T0*%
_class
loc:@layer_1/Variable_1*
validate_shape(*
_output_shapes
: 

layer_1/Variable_1/readIdentitylayer_1/Variable_1*
_output_shapes
: *
T0*%
_class
loc:@layer_1/Variable_1
ш
layer_1/Conv2DConv2DIteratorGetNextlayer_1/Variable/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
u
layer_1/addAddlayer_1/Conv2Dlayer_1/Variable_1/read*
T0*/
_output_shapes
:џџџџџџџџџ 
[
layer_1/ReluRelulayer_1/add*/
_output_shapes
:џџџџџџџџџ *
T0
Д
layer_1/MaxPoolMaxPoollayer_1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ 
w
layer_2/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"          @   
b
layer_2/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
layer_2/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
В
(layer_2/truncated_normal/TruncatedNormalTruncatedNormallayer_2/truncated_normal/shape*
dtype0*&
_output_shapes
: @*
seed2 *

seed *
T0

layer_2/truncated_normal/mulMul(layer_2/truncated_normal/TruncatedNormallayer_2/truncated_normal/stddev*&
_output_shapes
: @*
T0

layer_2/truncated_normalAddlayer_2/truncated_normal/mullayer_2/truncated_normal/mean*
T0*&
_output_shapes
: @

layer_2/Variable
VariableV2*
dtype0*&
_output_shapes
: @*
	container *
shape: @*
shared_name 
Ь
layer_2/Variable/AssignAssignlayer_2/Variablelayer_2/truncated_normal*
use_locking(*
T0*#
_class
loc:@layer_2/Variable*
validate_shape(*&
_output_shapes
: @

layer_2/Variable/readIdentitylayer_2/Variable*
T0*#
_class
loc:@layer_2/Variable*&
_output_shapes
: @
Z
layer_2/ConstConst*
valueB@*ЭЬЬ=*
dtype0*
_output_shapes
:@
~
layer_2/Variable_1
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Л
layer_2/Variable_1/AssignAssignlayer_2/Variable_1layer_2/Const*
use_locking(*
T0*%
_class
loc:@layer_2/Variable_1*
validate_shape(*
_output_shapes
:@

layer_2/Variable_1/readIdentitylayer_2/Variable_1*
_output_shapes
:@*
T0*%
_class
loc:@layer_2/Variable_1
ш
layer_2/Conv2DConv2Dlayer_1/MaxPoollayer_2/Variable/read*/
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
u
layer_2/addAddlayer_2/Conv2Dlayer_2/Variable_1/read*
T0*/
_output_shapes
:џџџџџџџџџ@
[
layer_2/ReluRelulayer_2/add*
T0*/
_output_shapes
:џџџџџџџџџ@
Д
layer_2/MaxPoolMaxPoollayer_2/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
T0
q
 fc_layer1/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"@     
d
fc_layer1/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
f
!fc_layer1/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
А
*fc_layer1/truncated_normal/TruncatedNormalTruncatedNormal fc_layer1/truncated_normal/shape*
dtype0* 
_output_shapes
:
Р*
seed2 *

seed *
T0

fc_layer1/truncated_normal/mulMul*fc_layer1/truncated_normal/TruncatedNormal!fc_layer1/truncated_normal/stddev* 
_output_shapes
:
Р*
T0

fc_layer1/truncated_normalAddfc_layer1/truncated_normal/mulfc_layer1/truncated_normal/mean*
T0* 
_output_shapes
:
Р

fc_layer1/Variable
VariableV2*
shared_name *
dtype0* 
_output_shapes
:
Р*
	container *
shape:
Р
Ю
fc_layer1/Variable/AssignAssignfc_layer1/Variablefc_layer1/truncated_normal*
validate_shape(* 
_output_shapes
:
Р*
use_locking(*
T0*%
_class
loc:@fc_layer1/Variable

fc_layer1/Variable/readIdentityfc_layer1/Variable*
T0*%
_class
loc:@fc_layer1/Variable* 
_output_shapes
:
Р
^
fc_layer1/ConstConst*
valueB*ЭЬЬ=*
dtype0*
_output_shapes	
:

fc_layer1/Variable_1
VariableV2*
dtype0*
_output_shapes	
:*
	container *
shape:*
shared_name 
Ф
fc_layer1/Variable_1/AssignAssignfc_layer1/Variable_1fc_layer1/Const*
use_locking(*
T0*'
_class
loc:@fc_layer1/Variable_1*
validate_shape(*
_output_shapes	
:

fc_layer1/Variable_1/readIdentityfc_layer1/Variable_1*
_output_shapes	
:*
T0*'
_class
loc:@fc_layer1/Variable_1
h
fc_layer1/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ@  

fc_layer1/ReshapeReshapelayer_2/MaxPoolfc_layer1/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџР

fc_layer1/MatMulMatMulfc_layer1/Reshapefc_layer1/Variable/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0
t
fc_layer1/addAddfc_layer1/MatMulfc_layer1/Variable_1/read*
T0*(
_output_shapes
:џџџџџџџџџ
X
fc_layer1/ReluRelufc_layer1/add*(
_output_shapes
:џџџџџџџџџ*
T0
t
#read_out_fc2/truncated_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
g
"read_out_fc2/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$read_out_fc2/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Е
-read_out_fc2/truncated_normal/TruncatedNormalTruncatedNormal#read_out_fc2/truncated_normal/shape*
T0*
dtype0*
_output_shapes
:	
*
seed2 *

seed 
Ї
!read_out_fc2/truncated_normal/mulMul-read_out_fc2/truncated_normal/TruncatedNormal$read_out_fc2/truncated_normal/stddev*
T0*
_output_shapes
:	


read_out_fc2/truncated_normalAdd!read_out_fc2/truncated_normal/mul"read_out_fc2/truncated_normal/mean*
_output_shapes
:	
*
T0

read_out_fc2/Variable
VariableV2*
dtype0*
_output_shapes
:	
*
	container *
shape:	
*
shared_name 
й
read_out_fc2/Variable/AssignAssignread_out_fc2/Variableread_out_fc2/truncated_normal*
use_locking(*
T0*(
_class
loc:@read_out_fc2/Variable*
validate_shape(*
_output_shapes
:	


read_out_fc2/Variable/readIdentityread_out_fc2/Variable*
T0*(
_class
loc:@read_out_fc2/Variable*
_output_shapes
:	

_
read_out_fc2/ConstConst*
dtype0*
_output_shapes
:
*
valueB
*ЭЬЬ=

read_out_fc2/Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes
:
*
	container *
shape:

Я
read_out_fc2/Variable_1/AssignAssignread_out_fc2/Variable_1read_out_fc2/Const*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0**
_class 
loc:@read_out_fc2/Variable_1

read_out_fc2/Variable_1/readIdentityread_out_fc2/Variable_1*
T0**
_class 
loc:@read_out_fc2/Variable_1*
_output_shapes
:

Ё
read_out_fc2/MatMulMatMulfc_layer1/Reluread_out_fc2/Variable/read*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
transpose_b( *
T0
|
read_out_fc2/addAddread_out_fc2/MatMulread_out_fc2/Variable_1/read*
T0*'
_output_shapes
:џџџџџџџџџ

z
)SparseSoftmaxCrossEntropyWithLogits/ShapeShapeIteratorGetNext:1*
_output_shapes
:*
T0*
out_type0
у
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsread_out_fc2/addIteratorGetNext:1*6
_output_shapes$
":џџџџџџџџџ:џџџџџџџџџ
*
Tlabels0*
T0
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 

MeanMeanGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

ArgMaxArgMaxread_out_fc2/addArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
Q
CastCastArgMax*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
U
EqualEqualCastIteratorGetNext:1*#
_output_shapes
:џџџџџџџџџ*
T0
S
ToFloatCastEqual*#
_output_shapes
:џџџџџџџџџ*

DstT0*

SrcT0


acc_op/total/Initializer/zerosConst*
_class
loc:@acc_op/total*
valueB
 *    *
dtype0*
_output_shapes
: 

acc_op/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@acc_op/total*
	container *
shape: 
Ж
acc_op/total/AssignAssignacc_op/totalacc_op/total/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@acc_op/total
m
acc_op/total/readIdentityacc_op/total*
T0*
_class
loc:@acc_op/total*
_output_shapes
: 

acc_op/count/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class
loc:@acc_op/count*
valueB
 *    

acc_op/count
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@acc_op/count*
	container *
shape: 
Ж
acc_op/count/AssignAssignacc_op/countacc_op/count/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@acc_op/count
m
acc_op/count/readIdentityacc_op/count*
_output_shapes
: *
T0*
_class
loc:@acc_op/count
M
acc_op/SizeSizeToFloat*
_output_shapes
: *
T0*
out_type0
U
acc_op/ToFloat_1Castacc_op/Size*

SrcT0*
_output_shapes
: *

DstT0
V
acc_op/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
f

acc_op/SumSumToFloatacc_op/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

acc_op/AssignAdd	AssignAddacc_op/total
acc_op/Sum*
use_locking( *
T0*
_class
loc:@acc_op/total*
_output_shapes
: 

acc_op/AssignAdd_1	AssignAddacc_op/countacc_op/ToFloat_1^ToFloat*
use_locking( *
T0*
_class
loc:@acc_op/count*
_output_shapes
: 
`
acc_op/truedivRealDivacc_op/total/readacc_op/count/read*
_output_shapes
: *
T0
V
acc_op/zeros_likeConst*
dtype0*
_output_shapes
: *
valueB
 *    
`
acc_op/GreaterGreateracc_op/count/readacc_op/zeros_like*
T0*
_output_shapes
: 
j
acc_op/valueSelectacc_op/Greateracc_op/truedivacc_op/zeros_like*
T0*
_output_shapes
: 
b
acc_op/truediv_1RealDivacc_op/AssignAddacc_op/AssignAdd_1*
T0*
_output_shapes
: 
X
acc_op/zeros_like_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
e
acc_op/Greater_1Greateracc_op/AssignAdd_1acc_op/zeros_like_1*
T0*
_output_shapes
: 
t
acc_op/update_opSelectacc_op/Greater_1acc_op/truediv_1acc_op/zeros_like_1*
T0*
_output_shapes
: 
V
accuracy/tagsConst*
valueB Baccuracy*
dtype0*
_output_shapes
: 
[
accuracyScalarSummaryaccuracy/tagsacc_op/update_op*
_output_shapes
: *
T0
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
 
gradients/Mean_grad/ShapeShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:џџџџџџџџџ
Ђ
gradients/Mean_grad/Shape_1ShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:џџџџџџџџџ

­
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:џџџџџџџџџ
*Д
messageЈЅCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
А
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
џџџџџџџџџ
Б
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Mean_grad/truedivegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*'
_output_shapes
:џџџџџџџџџ*

Tdim0*
T0
о
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*'
_output_shapes
:џџџџџџџџџ

x
%gradients/read_out_fc2/add_grad/ShapeShaperead_out_fc2/MatMul*
_output_shapes
:*
T0*
out_type0
q
'gradients/read_out_fc2/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
л
5gradients/read_out_fc2/add_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/read_out_fc2/add_grad/Shape'gradients/read_out_fc2/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
§
#gradients/read_out_fc2/add_grad/SumSumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul5gradients/read_out_fc2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
О
'gradients/read_out_fc2/add_grad/ReshapeReshape#gradients/read_out_fc2/add_grad/Sum%gradients/read_out_fc2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:џџџџџџџџџ


%gradients/read_out_fc2/add_grad/Sum_1SumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul7gradients/read_out_fc2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
З
)gradients/read_out_fc2/add_grad/Reshape_1Reshape%gradients/read_out_fc2/add_grad/Sum_1'gradients/read_out_fc2/add_grad/Shape_1*
_output_shapes
:
*
T0*
Tshape0

0gradients/read_out_fc2/add_grad/tuple/group_depsNoOp(^gradients/read_out_fc2/add_grad/Reshape*^gradients/read_out_fc2/add_grad/Reshape_1

8gradients/read_out_fc2/add_grad/tuple/control_dependencyIdentity'gradients/read_out_fc2/add_grad/Reshape1^gradients/read_out_fc2/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/read_out_fc2/add_grad/Reshape*'
_output_shapes
:џџџџџџџџџ


:gradients/read_out_fc2/add_grad/tuple/control_dependency_1Identity)gradients/read_out_fc2/add_grad/Reshape_11^gradients/read_out_fc2/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/read_out_fc2/add_grad/Reshape_1*
_output_shapes
:

т
)gradients/read_out_fc2/MatMul_grad/MatMulMatMul8gradients/read_out_fc2/add_grad/tuple/control_dependencyread_out_fc2/Variable/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
Я
+gradients/read_out_fc2/MatMul_grad/MatMul_1MatMulfc_layer1/Relu8gradients/read_out_fc2/add_grad/tuple/control_dependency*
T0*
_output_shapes
:	
*
transpose_a(*
transpose_b( 

3gradients/read_out_fc2/MatMul_grad/tuple/group_depsNoOp*^gradients/read_out_fc2/MatMul_grad/MatMul,^gradients/read_out_fc2/MatMul_grad/MatMul_1

;gradients/read_out_fc2/MatMul_grad/tuple/control_dependencyIdentity)gradients/read_out_fc2/MatMul_grad/MatMul4^gradients/read_out_fc2/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*<
_class2
0.loc:@gradients/read_out_fc2/MatMul_grad/MatMul

=gradients/read_out_fc2/MatMul_grad/tuple/control_dependency_1Identity+gradients/read_out_fc2/MatMul_grad/MatMul_14^gradients/read_out_fc2/MatMul_grad/tuple/group_deps*
_output_shapes
:	
*
T0*>
_class4
20loc:@gradients/read_out_fc2/MatMul_grad/MatMul_1
В
&gradients/fc_layer1/Relu_grad/ReluGradReluGrad;gradients/read_out_fc2/MatMul_grad/tuple/control_dependencyfc_layer1/Relu*
T0*(
_output_shapes
:џџџџџџџџџ
r
"gradients/fc_layer1/add_grad/ShapeShapefc_layer1/MatMul*
_output_shapes
:*
T0*
out_type0
o
$gradients/fc_layer1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
в
2gradients/fc_layer1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/fc_layer1/add_grad/Shape$gradients/fc_layer1/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
У
 gradients/fc_layer1/add_grad/SumSum&gradients/fc_layer1/Relu_grad/ReluGrad2gradients/fc_layer1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ж
$gradients/fc_layer1/add_grad/ReshapeReshape gradients/fc_layer1/add_grad/Sum"gradients/fc_layer1/add_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ч
"gradients/fc_layer1/add_grad/Sum_1Sum&gradients/fc_layer1/Relu_grad/ReluGrad4gradients/fc_layer1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Џ
&gradients/fc_layer1/add_grad/Reshape_1Reshape"gradients/fc_layer1/add_grad/Sum_1$gradients/fc_layer1/add_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

-gradients/fc_layer1/add_grad/tuple/group_depsNoOp%^gradients/fc_layer1/add_grad/Reshape'^gradients/fc_layer1/add_grad/Reshape_1

5gradients/fc_layer1/add_grad/tuple/control_dependencyIdentity$gradients/fc_layer1/add_grad/Reshape.^gradients/fc_layer1/add_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@gradients/fc_layer1/add_grad/Reshape
ќ
7gradients/fc_layer1/add_grad/tuple/control_dependency_1Identity&gradients/fc_layer1/add_grad/Reshape_1.^gradients/fc_layer1/add_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/fc_layer1/add_grad/Reshape_1*
_output_shapes	
:
й
&gradients/fc_layer1/MatMul_grad/MatMulMatMul5gradients/fc_layer1/add_grad/tuple/control_dependencyfc_layer1/Variable/read*(
_output_shapes
:џџџџџџџџџР*
transpose_a( *
transpose_b(*
T0
Э
(gradients/fc_layer1/MatMul_grad/MatMul_1MatMulfc_layer1/Reshape5gradients/fc_layer1/add_grad/tuple/control_dependency* 
_output_shapes
:
Р*
transpose_a(*
transpose_b( *
T0

0gradients/fc_layer1/MatMul_grad/tuple/group_depsNoOp'^gradients/fc_layer1/MatMul_grad/MatMul)^gradients/fc_layer1/MatMul_grad/MatMul_1

8gradients/fc_layer1/MatMul_grad/tuple/control_dependencyIdentity&gradients/fc_layer1/MatMul_grad/MatMul1^gradients/fc_layer1/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџР*
T0*9
_class/
-+loc:@gradients/fc_layer1/MatMul_grad/MatMul

:gradients/fc_layer1/MatMul_grad/tuple/control_dependency_1Identity(gradients/fc_layer1/MatMul_grad/MatMul_11^gradients/fc_layer1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/fc_layer1/MatMul_grad/MatMul_1* 
_output_shapes
:
Р
u
&gradients/fc_layer1/Reshape_grad/ShapeShapelayer_2/MaxPool*
T0*
out_type0*
_output_shapes
:
н
(gradients/fc_layer1/Reshape_grad/ReshapeReshape8gradients/fc_layer1/MatMul_grad/tuple/control_dependency&gradients/fc_layer1/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ@

*gradients/layer_2/MaxPool_grad/MaxPoolGradMaxPoolGradlayer_2/Relulayer_2/MaxPool(gradients/fc_layer1/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
T0*
data_formatNHWC*
strides

Є
$gradients/layer_2/Relu_grad/ReluGradReluGrad*gradients/layer_2/MaxPool_grad/MaxPoolGradlayer_2/Relu*/
_output_shapes
:џџџџџџџџџ@*
T0
n
 gradients/layer_2/add_grad/ShapeShapelayer_2/Conv2D*
T0*
out_type0*
_output_shapes
:
l
"gradients/layer_2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@
Ь
0gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/layer_2/add_grad/Shape"gradients/layer_2/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Н
gradients/layer_2/add_grad/SumSum$gradients/layer_2/Relu_grad/ReluGrad0gradients/layer_2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
"gradients/layer_2/add_grad/ReshapeReshapegradients/layer_2/add_grad/Sum gradients/layer_2/add_grad/Shape*/
_output_shapes
:џџџџџџџџџ@*
T0*
Tshape0
С
 gradients/layer_2/add_grad/Sum_1Sum$gradients/layer_2/Relu_grad/ReluGrad2gradients/layer_2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ј
$gradients/layer_2/add_grad/Reshape_1Reshape gradients/layer_2/add_grad/Sum_1"gradients/layer_2/add_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0

+gradients/layer_2/add_grad/tuple/group_depsNoOp#^gradients/layer_2/add_grad/Reshape%^gradients/layer_2/add_grad/Reshape_1

3gradients/layer_2/add_grad/tuple/control_dependencyIdentity"gradients/layer_2/add_grad/Reshape,^gradients/layer_2/add_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ@*
T0*5
_class+
)'loc:@gradients/layer_2/add_grad/Reshape
ѓ
5gradients/layer_2/add_grad/tuple/control_dependency_1Identity$gradients/layer_2/add_grad/Reshape_1,^gradients/layer_2/add_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:@

$gradients/layer_2/Conv2D_grad/ShapeNShapeNlayer_1/MaxPoollayer_2/Variable/read*
T0*
out_type0*
N* 
_output_shapes
::
|
#gradients/layer_2/Conv2D_grad/ConstConst*%
valueB"          @   *
dtype0*
_output_shapes
:
§
1gradients/layer_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/layer_2/Conv2D_grad/ShapeNlayer_2/Variable/read3gradients/layer_2/add_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
д
2gradients/layer_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterlayer_1/MaxPool#gradients/layer_2/Conv2D_grad/Const3gradients/layer_2/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: @

.gradients/layer_2/Conv2D_grad/tuple/group_depsNoOp3^gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter2^gradients/layer_2/Conv2D_grad/Conv2DBackpropInput
І
6gradients/layer_2/Conv2D_grad/tuple/control_dependencyIdentity1gradients/layer_2/Conv2D_grad/Conv2DBackpropInput/^gradients/layer_2/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/layer_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ 
Ё
8gradients/layer_2/Conv2D_grad/tuple/control_dependency_1Identity2gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter/^gradients/layer_2/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: @

*gradients/layer_1/MaxPool_grad/MaxPoolGradMaxPoolGradlayer_1/Relulayer_1/MaxPool6gradients/layer_2/Conv2D_grad/tuple/control_dependency*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
T0
Є
$gradients/layer_1/Relu_grad/ReluGradReluGrad*gradients/layer_1/MaxPool_grad/MaxPoolGradlayer_1/Relu*/
_output_shapes
:џџџџџџџџџ *
T0
n
 gradients/layer_1/add_grad/ShapeShapelayer_1/Conv2D*
T0*
out_type0*
_output_shapes
:
l
"gradients/layer_1/add_grad/Shape_1Const*
valueB: *
dtype0*
_output_shapes
:
Ь
0gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/layer_1/add_grad/Shape"gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
gradients/layer_1/add_grad/SumSum$gradients/layer_1/Relu_grad/ReluGrad0gradients/layer_1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
З
"gradients/layer_1/add_grad/ReshapeReshapegradients/layer_1/add_grad/Sum gradients/layer_1/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ 
С
 gradients/layer_1/add_grad/Sum_1Sum$gradients/layer_1/Relu_grad/ReluGrad2gradients/layer_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ј
$gradients/layer_1/add_grad/Reshape_1Reshape gradients/layer_1/add_grad/Sum_1"gradients/layer_1/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

+gradients/layer_1/add_grad/tuple/group_depsNoOp#^gradients/layer_1/add_grad/Reshape%^gradients/layer_1/add_grad/Reshape_1

3gradients/layer_1/add_grad/tuple/control_dependencyIdentity"gradients/layer_1/add_grad/Reshape,^gradients/layer_1/add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/layer_1/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџ 
ѓ
5gradients/layer_1/add_grad/tuple/control_dependency_1Identity$gradients/layer_1/add_grad/Reshape_1,^gradients/layer_1/add_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/layer_1/add_grad/Reshape_1

$gradients/layer_1/Conv2D_grad/ShapeNShapeNIteratorGetNextlayer_1/Variable/read*
N* 
_output_shapes
::*
T0*
out_type0
|
#gradients/layer_1/Conv2D_grad/ConstConst*
dtype0*
_output_shapes
:*%
valueB"             
§
1gradients/layer_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/layer_1/Conv2D_grad/ShapeNlayer_1/Variable/read3gradients/layer_1/add_grad/tuple/control_dependency*
paddingSAME*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
д
2gradients/layer_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterIteratorGetNext#gradients/layer_1/Conv2D_grad/Const3gradients/layer_1/add_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

.gradients/layer_1/Conv2D_grad/tuple/group_depsNoOp3^gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter2^gradients/layer_1/Conv2D_grad/Conv2DBackpropInput
І
6gradients/layer_1/Conv2D_grad/tuple/control_dependencyIdentity1gradients/layer_1/Conv2D_grad/Conv2DBackpropInput/^gradients/layer_1/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/layer_1/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ
Ё
8gradients/layer_1/Conv2D_grad/tuple/control_dependency_1Identity2gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter/^gradients/layer_1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
b
GradientDescent/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
Ј
<GradientDescent/update_layer_1/Variable/ApplyGradientDescentApplyGradientDescentlayer_1/VariableGradientDescent/learning_rate8gradients/layer_1/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
: *
use_locking( *
T0*#
_class
loc:@layer_1/Variable

>GradientDescent/update_layer_1/Variable_1/ApplyGradientDescentApplyGradientDescentlayer_1/Variable_1GradientDescent/learning_rate5gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@layer_1/Variable_1*
_output_shapes
: 
Ј
<GradientDescent/update_layer_2/Variable/ApplyGradientDescentApplyGradientDescentlayer_2/VariableGradientDescent/learning_rate8gradients/layer_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_2/Variable*&
_output_shapes
: @

>GradientDescent/update_layer_2/Variable_1/ApplyGradientDescentApplyGradientDescentlayer_2/Variable_1GradientDescent/learning_rate5gradients/layer_2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@layer_2/Variable_1*
_output_shapes
:@
Њ
>GradientDescent/update_fc_layer1/Variable/ApplyGradientDescentApplyGradientDescentfc_layer1/VariableGradientDescent/learning_rate:gradients/fc_layer1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@fc_layer1/Variable* 
_output_shapes
:
Р
Ј
@GradientDescent/update_fc_layer1/Variable_1/ApplyGradientDescentApplyGradientDescentfc_layer1/Variable_1GradientDescent/learning_rate7gradients/fc_layer1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@fc_layer1/Variable_1*
_output_shapes	
:
Е
AGradientDescent/update_read_out_fc2/Variable/ApplyGradientDescentApplyGradientDescentread_out_fc2/VariableGradientDescent/learning_rate=gradients/read_out_fc2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@read_out_fc2/Variable*
_output_shapes
:	

Г
CGradientDescent/update_read_out_fc2/Variable_1/ApplyGradientDescentApplyGradientDescentread_out_fc2/Variable_1GradientDescent/learning_rate:gradients/read_out_fc2/add_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@read_out_fc2/Variable_1*
_output_shapes
:

Ќ
GradientDescent/updateNoOp?^GradientDescent/update_fc_layer1/Variable/ApplyGradientDescentA^GradientDescent/update_fc_layer1/Variable_1/ApplyGradientDescent=^GradientDescent/update_layer_1/Variable/ApplyGradientDescent?^GradientDescent/update_layer_1/Variable_1/ApplyGradientDescent=^GradientDescent/update_layer_2/Variable/ApplyGradientDescent?^GradientDescent/update_layer_2/Variable_1/ApplyGradientDescentB^GradientDescent/update_read_out_fc2/Variable/ApplyGradientDescentD^GradientDescent/update_read_out_fc2/Variable_1/ApplyGradientDescent

GradientDescent/valueConst^GradientDescent/update*
_class
loc:@global_step*
value	B	 R*
dtype0	*
_output_shapes
: 

GradientDescent	AssignAddglobal_stepGradientDescent/value*
use_locking( *
T0	*
_class
loc:@global_step*
_output_shapes
: 
N
	loss/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bloss
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 

initNoOp^fc_layer1/Variable/Assign^fc_layer1/Variable_1/Assign^global_step/Assign^layer_1/Variable/Assign^layer_1/Variable_1/Assign^layer_2/Variable/Assign^layer_2/Variable_1/Assign^read_out_fc2/Variable/Assign^read_out_fc2/Variable_1/Assign

init_1NoOp
"

group_depsNoOp^init^init_1

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_output_shapes
: *
_class
loc:@global_step
Ћ
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedlayer_1/Variable*
dtype0*
_output_shapes
: *#
_class
loc:@layer_1/Variable
Џ
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedlayer_1/Variable_1*%
_class
loc:@layer_1/Variable_1*
dtype0*
_output_shapes
: 
Ћ
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedlayer_2/Variable*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/Variable
Џ
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedlayer_2/Variable_1*%
_class
loc:@layer_2/Variable_1*
dtype0*
_output_shapes
: 
Џ
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedfc_layer1/Variable*%
_class
loc:@fc_layer1/Variable*
dtype0*
_output_shapes
: 
Г
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializedfc_layer1/Variable_1*'
_class
loc:@fc_layer1/Variable_1*
dtype0*
_output_shapes
: 
Е
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializedread_out_fc2/Variable*
dtype0*
_output_shapes
: *(
_class
loc:@read_out_fc2/Variable
Й
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializedread_out_fc2/Variable_1*
dtype0*
_output_shapes
: **
_class 
loc:@read_out_fc2/Variable_1
Ѓ
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedacc_op/total*
dtype0*
_output_shapes
: *
_class
loc:@acc_op/total
Є
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedacc_op/count*
dtype0*
_output_shapes
: *
_class
loc:@acc_op/count
м
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_10"/device:CPU:0*
N*
_output_shapes
:*
T0
*

axis 

)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
Ь
$report_uninitialized_variables/ConstConst"/device:CPU:0*
dtype0*
_output_shapes
:*ф
valueкBзBglobal_stepBlayer_1/VariableBlayer_1/Variable_1Blayer_2/VariableBlayer_2/Variable_1Bfc_layer1/VariableBfc_layer1/Variable_1Bread_out_fc2/VariableBread_out_fc2/Variable_1Bacc_op/totalBacc_op/count

1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
ш
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
№
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask 

3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
№
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
end_mask*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask 
О
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 

7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
ї
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*
N*
_output_shapes
:*

Tidx0*
T0
к
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
_output_shapes
:*
T0*
Tshape0

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
џџџџџџџџџ
ъ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:
В
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:џџџџџџџџџ*
T0

Х
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
*
T0	

9report_uninitialized_variables/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Х
4report_uninitialized_variables/boolean_mask/GatherV2GatherV23report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze9report_uninitialized_variables/boolean_mask/GatherV2/axis"/device:CPU:0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Taxis0
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB 
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
О
concatConcatV24report_uninitialized_variables/boolean_mask/GatherV2$report_uninitialized_resources/Constconcat/axis*
T0*
N*#
_output_shapes
:џџџџџџџџџ*

Tidx0
Ё
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
­
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedlayer_1/Variable*
dtype0*
_output_shapes
: *#
_class
loc:@layer_1/Variable
Б
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedlayer_1/Variable_1*%
_class
loc:@layer_1/Variable_1*
dtype0*
_output_shapes
: 
­
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializedlayer_2/Variable*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/Variable
Б
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedlayer_2/Variable_1*
dtype0*
_output_shapes
: *%
_class
loc:@layer_2/Variable_1
Б
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializedfc_layer1/Variable*%
_class
loc:@fc_layer1/Variable*
dtype0*
_output_shapes
: 
Е
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializedfc_layer1/Variable_1*'
_class
loc:@fc_layer1/Variable_1*
dtype0*
_output_shapes
: 
З
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializedread_out_fc2/Variable*
dtype0*
_output_shapes
: *(
_class
loc:@read_out_fc2/Variable
Л
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializedread_out_fc2/Variable_1**
_class 
loc:@read_out_fc2/Variable_1*
dtype0*
_output_shapes
: 
џ
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_8"/device:CPU:0*
N	*
_output_shapes
:	*
T0
*

axis 

+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:	
В
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*
dtype0*
_output_shapes
:	*Ш
valueОBЛ	Bglobal_stepBlayer_1/VariableBlayer_1/Variable_1Blayer_2/VariableBlayer_2/Variable_1Bfc_layer1/VariableBfc_layer1/Variable_1Bread_out_fc2/VariableBread_out_fc2/Variable_1

3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:	

Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
ђ
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask 

Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
valueB:	*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 

5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:	

Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
Index0*
T0
Т
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 

9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 

4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
T0*
N*
_output_shapes
:*

Tidx0
р
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
_output_shapes
:	*
T0*
Tshape0

=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
№
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
T0
*
Tshape0*
_output_shapes
:	
Ж
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:џџџџџџџџџ*
T0

Щ
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
*
T0	

;report_uninitialized_variables_1/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Э
6report_uninitialized_variables_1/boolean_mask/GatherV2GatherV25report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze;report_uninitialized_variables_1/boolean_mask/GatherV2/axis"/device:CPU:0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Taxis0
:
init_2NoOp^acc_op/count/Assign^acc_op/total/Assign

init_all_tablesNoOp

init_3NoOp
8
group_deps_1NoOp^init_2^init_3^init_all_tables
S
Merge/MergeSummaryMergeSummaryaccuracyloss*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_6f190f302adc4398a12d5894a00594e7/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Є
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:	*Ш
valueОBЛ	Bfc_layer1/VariableBfc_layer1/Variable_1Bglobal_stepBlayer_1/VariableBlayer_1/Variable_1Blayer_2/VariableBlayer_2/Variable_1Bread_out_fc2/VariableBread_out_fc2/Variable_1

save/SaveV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
О
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesfc_layer1/Variablefc_layer1/Variable_1global_steplayer_1/Variablelayer_1/Variable_1layer_2/Variablelayer_2/Variable_1read_out_fc2/Variableread_out_fc2/Variable_1"/device:CPU:0*
dtypes
2		
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
Ї
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:	*Ш
valueОBЛ	Bfc_layer1/VariableBfc_layer1/Variable_1Bglobal_stepBlayer_1/VariableBlayer_1/Variable_1Blayer_2/VariableBlayer_2/Variable_1Bread_out_fc2/VariableBread_out_fc2/Variable_1

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:	*%
valueB	B B B B B B B B B 
Ч
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		
Д
save/AssignAssignfc_layer1/Variablesave/RestoreV2*
validate_shape(* 
_output_shapes
:
Р*
use_locking(*
T0*%
_class
loc:@fc_layer1/Variable
З
save/Assign_1Assignfc_layer1/Variable_1save/RestoreV2:1*
use_locking(*
T0*'
_class
loc:@fc_layer1/Variable_1*
validate_shape(*
_output_shapes	
:
 
save/Assign_2Assignglobal_stepsave/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
К
save/Assign_3Assignlayer_1/Variablesave/RestoreV2:3*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*#
_class
loc:@layer_1/Variable
В
save/Assign_4Assignlayer_1/Variable_1save/RestoreV2:4*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@layer_1/Variable_1
К
save/Assign_5Assignlayer_2/Variablesave/RestoreV2:5*
validate_shape(*&
_output_shapes
: @*
use_locking(*
T0*#
_class
loc:@layer_2/Variable
В
save/Assign_6Assignlayer_2/Variable_1save/RestoreV2:6*
use_locking(*
T0*%
_class
loc:@layer_2/Variable_1*
validate_shape(*
_output_shapes
:@
Н
save/Assign_7Assignread_out_fc2/Variablesave/RestoreV2:7*
use_locking(*
T0*(
_class
loc:@read_out_fc2/Variable*
validate_shape(*
_output_shapes
:	

М
save/Assign_8Assignread_out_fc2/Variable_1save/RestoreV2:8*
use_locking(*
T0**
_class 
loc:@read_out_fc2/Variable_1*
validate_shape(*
_output_shapes
:

Ј
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shardУ+
ю

tf_map_func_6yLLF6rv0aU
arg0
resize_images_squeeze

cond_merge25A wrapper for Defun that facilitates shape inference./
ConstConst*
dtype0*
value	B B/2
packedPackarg0*
N*
T0*

axis M
StringSplitStringSplitpacked:output:0Const:output:0*

skip_empty(J
strided_slice/stackConst*
valueB:
ўџџџџџџџџ*
dtype0L
strided_slice/stack_1Const*
valueB:
џџџџџџџџџ*
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0
strided_sliceStridedSliceStringSplit:values:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
end_mask *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 4
Equal/yConst*
dtype0*
valueB
 BdogsA
EqualEqualstrided_slice:output:0Equal/y:output:0*
T04
cond/SwitchSwitch	Equal:z:0	Equal:z:0*
T0
=
cond/switch_tIdentitycond/Switch:output_true:0*
T0
>
cond/switch_fIdentitycond/Switch:output_false:0*
T0
,
cond/pred_idIdentity	Equal:z:0*
T0
D

cond/ConstConst^cond/switch_t*
value	B : *
dtype0F
cond/Const_1Const^cond/switch_f*
dtype0*
value	B :Q

cond/MergeMergecond/Const_1:output:0cond/Const:output:0*
N*
T0
ReadFileReadFilearg0Ў

DecodeJpeg
DecodeJpegReadFile:contents:0*
fancy_upscaling(*
try_recover_truncated( *
ratio*

dct_method *
channels*
acceptable_fraction%  ?F
convert_image/CastCastDecodeJpeg:image:0*

SrcT0*

DstT0<
convert_image/yConst*
valueB
 *;*
dtype0O
convert_imageMulconvert_image/Cast:y:0convert_image/y:output:0*
T0F
resize_images/ExpandDims/dimConst*
dtype0*
value	B : u
resize_images/ExpandDims
ExpandDimsconvert_image:z:0%resize_images/ExpandDims/dim:output:0*

Tdim0*
T0G
resize_images/sizeConst*
dtype0*
valueB"      
resize_images/ResizeBilinearResizeBilinear!resize_images/ExpandDims:output:0resize_images/size:output:0*
align_corners( *
T0o
resize_images/SqueezeSqueeze-resize_images/ResizeBilinear:resized_images:0*
squeeze_dims
 *
T0"!

cond_mergecond/Merge:output:0"7
resize_images_squeezeresize_images/Squeeze:output:0
Я
3
_make_dataset_GUdGPfB2wf0
prefetchdatasetn
(TensorSliceDataset/MatchingFiles/patternConst*
dtype0*.
value%B# B/data/dogscats/train/**/*.jpgd
 TensorSliceDataset/MatchingFilesMatchingFiles1TensorSliceDataset/MatchingFiles/pattern:output:0
TensorSliceDatasetTensorSliceDataset,TensorSliceDataset/MatchingFiles:filenames:0*
output_shapes
: *
Toutput_types
2j
$ShuffleDataset/MatchingFiles/patternConst*.
value%B# B/data/dogscats/train/**/*.jpg*
dtype0\
ShuffleDataset/MatchingFilesMatchingFiles-ShuffleDataset/MatchingFiles/pattern:output:0`
ShuffleDataset/ShapeShape(ShuffleDataset/MatchingFiles:filenames:0*
T0*
out_type0	P
"ShuffleDataset/strided_slice/stackConst*
valueB: *
dtype0R
$ShuffleDataset/strided_slice/stack_1Const*
dtype0*
valueB:R
$ShuffleDataset/strided_slice/stack_2Const*
dtype0*
valueB:а
ShuffleDataset/strided_sliceStridedSliceShuffleDataset/Shape:output:0+ShuffleDataset/strided_slice/stack:output:0-ShuffleDataset/strided_slice/stack_1:output:0-ShuffleDataset/strided_slice/stack_2:output:0*
end_mask *
Index0*
T0	*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask B
ShuffleDataset/Maximum/yConst*
dtype0	*
value	B	 Rt
ShuffleDataset/MaximumMaximum%ShuffleDataset/strided_slice:output:0!ShuffleDataset/Maximum/y:output:0*
T0	=
ShuffleDataset/seedConst*
dtype0	*
value	B	 R >
ShuffleDataset/seed2Const*
value	B	 R *
dtype0	ф
ShuffleDatasetShuffleDatasetTensorSliceDataset:handle:0ShuffleDataset/Maximum:z:0ShuffleDataset/seed:output:0ShuffleDataset/seed2:output:0*
reshuffle_each_iteration(*
output_types
2*
output_shapes
: N
#ShuffleAndRepeatDataset/buffer_sizeConst*
dtype0	*
value
B	 R H
ShuffleAndRepeatDataset/seed_1Const*
dtype0	*
value	B	 R I
ShuffleAndRepeatDataset/seed2_1Const*
value	B	 R *
dtype0	G
ShuffleAndRepeatDataset/countConst*
dtype0	*
value	B	 R
Ђ
ShuffleAndRepeatDatasetShuffleAndRepeatDatasetShuffleDataset:handle:0,ShuffleAndRepeatDataset/buffer_size:output:0'ShuffleAndRepeatDataset/seed_1:output:0(ShuffleAndRepeatDataset/seed2_1:output:0&ShuffleAndRepeatDataset/count:output:0*
output_shapes
: *
output_types
2Ћ

MapDataset
MapDataset ShuffleAndRepeatDataset:handle:0*

Targuments
 *#
output_shapes
:: * 
fR
tf_map_func_6yLLF6rv0aU*
output_types
2A
BatchDataset/batch_sizeConst*
dtype0	*
value	B	 R@Њ
BatchDatasetBatchDatasetMapDataset:handle:0 BatchDataset/batch_size:output:0*
output_types
2*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџH
PrefetchDataset/buffer_size_1Const*
dtype0	*
value
B	 R И
PrefetchDatasetPrefetchDatasetBatchDataset:handle:0&PrefetchDataset/buffer_size_1:output:0*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*
output_types
2"+
prefetchdatasetPrefetchDataset:handle:0"­іPП3     шшџ	ќГЅ#>ЩжAJВч
Ќ33
:
Add
x"T
y"T
z"T"
Ttype:
2	

ApplyGradientDescent
var"T

alpha"T

delta"T
out"T" 
Ttype:
2	"
use_lockingbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
ь
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)


Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype
is_initialized
"
dtypetype

IteratorGetNext
iterator

components2output_types"
output_types
list(type)(0" 
output_shapeslist(shape)(0
C
IteratorToStringHandle
resource_handle
string_handle


LogicalNot
x

y

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
+
MatchingFiles
pattern
	filenames
д
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ю
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
Џ
OneShotIterator

handle"
dataset_factoryfunc"
output_types
list(type)(0" 
output_shapeslist(shape)(0"
	containerstring "
shared_namestring 
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
\
	RefSwitch
data"T
pred

output_false"T
output_true"T"	
Ttype
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	

#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype*1.8.02
b'unknown'К

global_step/Initializer/zerosConst*
dtype0	*
_output_shapes
: *
_class
loc:@global_step*
value	B	 R 

global_step
VariableV2*
shared_name *
_class
loc:@global_step*
	container *
shape: *
dtype0	*
_output_shapes
: 
В
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 

!global_step/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 

global_step/cond/SwitchSwitch!global_step/IsVariableInitialized!global_step/IsVariableInitialized*
_output_shapes
: : *
T0

a
global_step/cond/switch_tIdentityglobal_step/cond/Switch:1*
T0
*
_output_shapes
: 
_
global_step/cond/switch_fIdentityglobal_step/cond/Switch*
T0
*
_output_shapes
: 
h
global_step/cond/pred_idIdentity!global_step/IsVariableInitialized*
_output_shapes
: *
T0

b
global_step/cond/readIdentityglobal_step/cond/read/Switch:1*
T0	*
_output_shapes
: 

global_step/cond/read/Switch	RefSwitchglobal_stepglobal_step/cond/pred_id*
_output_shapes
: : *
T0	*
_class
loc:@global_step

global_step/cond/Switch_1Switchglobal_step/Initializer/zerosglobal_step/cond/pred_id*
_output_shapes
: : *
T0	*
_class
loc:@global_step
}
global_step/cond/MergeMergeglobal_step/cond/Switch_1global_step/cond/read*
T0	*
N*
_output_shapes
: : 
S
global_step/add/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
b
global_step/addAddglobal_step/cond/Mergeglobal_step/add/y*
_output_shapes
: *
T0	

MatchingFiles/patternConst"/device:CPU:0*
dtype0*
_output_shapes
: *.
value%B# B/data/dogscats/train/**/*.jpg
i
MatchingFilesMatchingFilesMatchingFiles/pattern"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ
a
ShapeShapeMatchingFiles"/device:CPU:0*
_output_shapes
:*
T0*
out_type0	
l
strided_slice/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 
n
strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
n
strided_slice/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2"/device:CPU:0*
T0	*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Z
	Maximum/yConst"/device:CPU:0*
value	B	 R*
dtype0	*
_output_shapes
: 
\
MaximumMaximumstrided_slice	Maximum/y"/device:CPU:0*
T0	*
_output_shapes
: 
U
seedConst"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
V
seed2Const"/device:CPU:0*
dtype0	*
_output_shapes
: *
value	B	 R 
]
buffer_sizeConst"/device:CPU:0*
value
B	 R *
dtype0	*
_output_shapes
: 
V
countConst"/device:CPU:0*
value	B	 R
*
dtype0	*
_output_shapes
: 
W
seed_1Const"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
X
seed2_1Const"/device:CPU:0*
value	B	 R *
dtype0	*
_output_shapes
: 
[

batch_sizeConst"/device:CPU:0*
value	B	 R@*
dtype0	*
_output_shapes
: 
_
buffer_size_1Const"/device:CPU:0*
dtype0	*
_output_shapes
: *
value
B	 R 
і
OneShotIteratorOneShotIterator"/device:CPU:0*
_output_shapes
: *0
dataset_factoryR
_make_dataset_GUdGPfB2wf0*
shared_name *=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*
	container *
output_types
2
h
IteratorToStringHandleIteratorToStringHandleOneShotIterator"/device:CPU:0*
_output_shapes
: 
й
IteratorGetNextIteratorGetNextOneShotIterator"/device:CPU:0*>
_output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*
output_types
2*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ
w
layer_1/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*%
valueB"             
b
layer_1/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
d
layer_1/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
В
(layer_1/truncated_normal/TruncatedNormalTruncatedNormallayer_1/truncated_normal/shape*
dtype0*&
_output_shapes
: *
seed2 *

seed *
T0

layer_1/truncated_normal/mulMul(layer_1/truncated_normal/TruncatedNormallayer_1/truncated_normal/stddev*&
_output_shapes
: *
T0

layer_1/truncated_normalAddlayer_1/truncated_normal/mullayer_1/truncated_normal/mean*&
_output_shapes
: *
T0

layer_1/Variable
VariableV2*
dtype0*&
_output_shapes
: *
	container *
shape: *
shared_name 
Ь
layer_1/Variable/AssignAssignlayer_1/Variablelayer_1/truncated_normal*
use_locking(*
T0*#
_class
loc:@layer_1/Variable*
validate_shape(*&
_output_shapes
: 

layer_1/Variable/readIdentitylayer_1/Variable*
T0*#
_class
loc:@layer_1/Variable*&
_output_shapes
: 
Z
layer_1/ConstConst*
valueB *ЭЬЬ=*
dtype0*
_output_shapes
: 
~
layer_1/Variable_1
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
Л
layer_1/Variable_1/AssignAssignlayer_1/Variable_1layer_1/Const*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@layer_1/Variable_1

layer_1/Variable_1/readIdentitylayer_1/Variable_1*
_output_shapes
: *
T0*%
_class
loc:@layer_1/Variable_1
ш
layer_1/Conv2DConv2DIteratorGetNextlayer_1/Variable/read*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
u
layer_1/addAddlayer_1/Conv2Dlayer_1/Variable_1/read*
T0*/
_output_shapes
:џџџџџџџџџ 
[
layer_1/ReluRelulayer_1/add*
T0*/
_output_shapes
:џџџџџџџџџ 
Д
layer_1/MaxPoolMaxPoollayer_1/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
T0
w
layer_2/truncated_normal/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
b
layer_2/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
d
layer_2/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
В
(layer_2/truncated_normal/TruncatedNormalTruncatedNormallayer_2/truncated_normal/shape*
T0*
dtype0*&
_output_shapes
: @*
seed2 *

seed 

layer_2/truncated_normal/mulMul(layer_2/truncated_normal/TruncatedNormallayer_2/truncated_normal/stddev*&
_output_shapes
: @*
T0

layer_2/truncated_normalAddlayer_2/truncated_normal/mullayer_2/truncated_normal/mean*
T0*&
_output_shapes
: @

layer_2/Variable
VariableV2*
shared_name *
dtype0*&
_output_shapes
: @*
	container *
shape: @
Ь
layer_2/Variable/AssignAssignlayer_2/Variablelayer_2/truncated_normal*
use_locking(*
T0*#
_class
loc:@layer_2/Variable*
validate_shape(*&
_output_shapes
: @

layer_2/Variable/readIdentitylayer_2/Variable*
T0*#
_class
loc:@layer_2/Variable*&
_output_shapes
: @
Z
layer_2/ConstConst*
dtype0*
_output_shapes
:@*
valueB@*ЭЬЬ=
~
layer_2/Variable_1
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
Л
layer_2/Variable_1/AssignAssignlayer_2/Variable_1layer_2/Const*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@layer_2/Variable_1

layer_2/Variable_1/readIdentitylayer_2/Variable_1*
_output_shapes
:@*
T0*%
_class
loc:@layer_2/Variable_1
ш
layer_2/Conv2DConv2Dlayer_1/MaxPoollayer_2/Variable/read*/
_output_shapes
:џџџџџџџџџ@*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
u
layer_2/addAddlayer_2/Conv2Dlayer_2/Variable_1/read*
T0*/
_output_shapes
:џџџџџџџџџ@
[
layer_2/ReluRelulayer_2/add*/
_output_shapes
:џџџџџџџџџ@*
T0
Д
layer_2/MaxPoolMaxPoollayer_2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@
q
 fc_layer1/truncated_normal/shapeConst*
valueB"@     *
dtype0*
_output_shapes
:
d
fc_layer1/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
f
!fc_layer1/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
А
*fc_layer1/truncated_normal/TruncatedNormalTruncatedNormal fc_layer1/truncated_normal/shape*
dtype0* 
_output_shapes
:
Р*
seed2 *

seed *
T0

fc_layer1/truncated_normal/mulMul*fc_layer1/truncated_normal/TruncatedNormal!fc_layer1/truncated_normal/stddev* 
_output_shapes
:
Р*
T0

fc_layer1/truncated_normalAddfc_layer1/truncated_normal/mulfc_layer1/truncated_normal/mean*
T0* 
_output_shapes
:
Р

fc_layer1/Variable
VariableV2*
dtype0* 
_output_shapes
:
Р*
	container *
shape:
Р*
shared_name 
Ю
fc_layer1/Variable/AssignAssignfc_layer1/Variablefc_layer1/truncated_normal*
validate_shape(* 
_output_shapes
:
Р*
use_locking(*
T0*%
_class
loc:@fc_layer1/Variable

fc_layer1/Variable/readIdentityfc_layer1/Variable* 
_output_shapes
:
Р*
T0*%
_class
loc:@fc_layer1/Variable
^
fc_layer1/ConstConst*
valueB*ЭЬЬ=*
dtype0*
_output_shapes	
:

fc_layer1/Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes	
:*
	container *
shape:
Ф
fc_layer1/Variable_1/AssignAssignfc_layer1/Variable_1fc_layer1/Const*
use_locking(*
T0*'
_class
loc:@fc_layer1/Variable_1*
validate_shape(*
_output_shapes	
:

fc_layer1/Variable_1/readIdentityfc_layer1/Variable_1*
_output_shapes	
:*
T0*'
_class
loc:@fc_layer1/Variable_1
h
fc_layer1/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ@  

fc_layer1/ReshapeReshapelayer_2/MaxPoolfc_layer1/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:џџџџџџџџџР

fc_layer1/MatMulMatMulfc_layer1/Reshapefc_layer1/Variable/read*
T0*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( 
t
fc_layer1/addAddfc_layer1/MatMulfc_layer1/Variable_1/read*
T0*(
_output_shapes
:џџџџџџџџџ
X
fc_layer1/ReluRelufc_layer1/add*(
_output_shapes
:џџџџџџџџџ*
T0
t
#read_out_fc2/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"   
   
g
"read_out_fc2/truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
i
$read_out_fc2/truncated_normal/stddevConst*
valueB
 *ЭЬЬ=*
dtype0*
_output_shapes
: 
Е
-read_out_fc2/truncated_normal/TruncatedNormalTruncatedNormal#read_out_fc2/truncated_normal/shape*
T0*
dtype0*
_output_shapes
:	
*
seed2 *

seed 
Ї
!read_out_fc2/truncated_normal/mulMul-read_out_fc2/truncated_normal/TruncatedNormal$read_out_fc2/truncated_normal/stddev*
T0*
_output_shapes
:	


read_out_fc2/truncated_normalAdd!read_out_fc2/truncated_normal/mul"read_out_fc2/truncated_normal/mean*
_output_shapes
:	
*
T0

read_out_fc2/Variable
VariableV2*
dtype0*
_output_shapes
:	
*
	container *
shape:	
*
shared_name 
й
read_out_fc2/Variable/AssignAssignread_out_fc2/Variableread_out_fc2/truncated_normal*
use_locking(*
T0*(
_class
loc:@read_out_fc2/Variable*
validate_shape(*
_output_shapes
:	


read_out_fc2/Variable/readIdentityread_out_fc2/Variable*
_output_shapes
:	
*
T0*(
_class
loc:@read_out_fc2/Variable
_
read_out_fc2/ConstConst*
valueB
*ЭЬЬ=*
dtype0*
_output_shapes
:


read_out_fc2/Variable_1
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 
Я
read_out_fc2/Variable_1/AssignAssignread_out_fc2/Variable_1read_out_fc2/Const*
use_locking(*
T0**
_class 
loc:@read_out_fc2/Variable_1*
validate_shape(*
_output_shapes
:


read_out_fc2/Variable_1/readIdentityread_out_fc2/Variable_1*
_output_shapes
:
*
T0**
_class 
loc:@read_out_fc2/Variable_1
Ё
read_out_fc2/MatMulMatMulfc_layer1/Reluread_out_fc2/Variable/read*
T0*'
_output_shapes
:џџџџџџџџџ
*
transpose_a( *
transpose_b( 
|
read_out_fc2/addAddread_out_fc2/MatMulread_out_fc2/Variable_1/read*'
_output_shapes
:џџџџџџџџџ
*
T0
z
)SparseSoftmaxCrossEntropyWithLogits/ShapeShapeIteratorGetNext:1*
_output_shapes
:*
T0*
out_type0
у
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsread_out_fc2/addIteratorGetNext:1*
T0*6
_output_shapes$
":џџџџџџџџџ:џџџџџџџџџ
*
Tlabels0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:

MeanMeanGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsConst*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :

ArgMaxArgMaxread_out_fc2/addArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:џџџџџџџџџ*

Tidx0
Q
CastCastArgMax*

SrcT0	*#
_output_shapes
:џџџџџџџџџ*

DstT0
U
EqualEqualCastIteratorGetNext:1*
T0*#
_output_shapes
:џџџџџџџџџ
S
ToFloatCastEqual*

SrcT0
*#
_output_shapes
:џџџџџџџџџ*

DstT0

acc_op/total/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class
loc:@acc_op/total*
valueB
 *    

acc_op/total
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@acc_op/total*
	container *
shape: 
Ж
acc_op/total/AssignAssignacc_op/totalacc_op/total/Initializer/zeros*
use_locking(*
T0*
_class
loc:@acc_op/total*
validate_shape(*
_output_shapes
: 
m
acc_op/total/readIdentityacc_op/total*
T0*
_class
loc:@acc_op/total*
_output_shapes
: 

acc_op/count/Initializer/zerosConst*
_class
loc:@acc_op/count*
valueB
 *    *
dtype0*
_output_shapes
: 

acc_op/count
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@acc_op/count*
	container *
shape: 
Ж
acc_op/count/AssignAssignacc_op/countacc_op/count/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@acc_op/count
m
acc_op/count/readIdentityacc_op/count*
_output_shapes
: *
T0*
_class
loc:@acc_op/count
M
acc_op/SizeSizeToFloat*
T0*
out_type0*
_output_shapes
: 
U
acc_op/ToFloat_1Castacc_op/Size*

SrcT0*
_output_shapes
: *

DstT0
V
acc_op/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
f

acc_op/SumSumToFloatacc_op/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0

acc_op/AssignAdd	AssignAddacc_op/total
acc_op/Sum*
use_locking( *
T0*
_class
loc:@acc_op/total*
_output_shapes
: 

acc_op/AssignAdd_1	AssignAddacc_op/countacc_op/ToFloat_1^ToFloat*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@acc_op/count
`
acc_op/truedivRealDivacc_op/total/readacc_op/count/read*
T0*
_output_shapes
: 
V
acc_op/zeros_likeConst*
dtype0*
_output_shapes
: *
valueB
 *    
`
acc_op/GreaterGreateracc_op/count/readacc_op/zeros_like*
_output_shapes
: *
T0
j
acc_op/valueSelectacc_op/Greateracc_op/truedivacc_op/zeros_like*
T0*
_output_shapes
: 
b
acc_op/truediv_1RealDivacc_op/AssignAddacc_op/AssignAdd_1*
_output_shapes
: *
T0
X
acc_op/zeros_like_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
e
acc_op/Greater_1Greateracc_op/AssignAdd_1acc_op/zeros_like_1*
_output_shapes
: *
T0
t
acc_op/update_opSelectacc_op/Greater_1acc_op/truediv_1acc_op/zeros_like_1*
_output_shapes
: *
T0
V
accuracy/tagsConst*
valueB Baccuracy*
dtype0*
_output_shapes
: 
[
accuracyScalarSummaryaccuracy/tagsacc_op/update_op*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
 
gradients/Mean_grad/ShapeShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
_output_shapes
:*
T0*
out_type0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:џџџџџџџџџ*

Tmultiples0*
T0
Ђ
gradients/Mean_grad/Shape_1ShapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
n
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:џџџџџџџџџ

gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*'
_output_shapes
:џџџџџџџџџ
*
T0
­
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*'
_output_shapes
:џџџџџџџџџ
*Д
messageЈЅCurrently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
А
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
џџџџџџџџџ*
dtype0*
_output_shapes
: 
Б
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Mean_grad/truedivegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
о
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*'
_output_shapes
:џџџџџџџџџ
*
T0
x
%gradients/read_out_fc2/add_grad/ShapeShaperead_out_fc2/MatMul*
_output_shapes
:*
T0*
out_type0
q
'gradients/read_out_fc2/add_grad/Shape_1Const*
valueB:
*
dtype0*
_output_shapes
:
л
5gradients/read_out_fc2/add_grad/BroadcastGradientArgsBroadcastGradientArgs%gradients/read_out_fc2/add_grad/Shape'gradients/read_out_fc2/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
§
#gradients/read_out_fc2/add_grad/SumSumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul5gradients/read_out_fc2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
О
'gradients/read_out_fc2/add_grad/ReshapeReshape#gradients/read_out_fc2/add_grad/Sum%gradients/read_out_fc2/add_grad/Shape*'
_output_shapes
:џџџџџџџџџ
*
T0*
Tshape0

%gradients/read_out_fc2/add_grad/Sum_1SumZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul7gradients/read_out_fc2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
З
)gradients/read_out_fc2/add_grad/Reshape_1Reshape%gradients/read_out_fc2/add_grad/Sum_1'gradients/read_out_fc2/add_grad/Shape_1*
_output_shapes
:
*
T0*
Tshape0

0gradients/read_out_fc2/add_grad/tuple/group_depsNoOp(^gradients/read_out_fc2/add_grad/Reshape*^gradients/read_out_fc2/add_grad/Reshape_1

8gradients/read_out_fc2/add_grad/tuple/control_dependencyIdentity'gradients/read_out_fc2/add_grad/Reshape1^gradients/read_out_fc2/add_grad/tuple/group_deps*'
_output_shapes
:џџџџџџџџџ
*
T0*:
_class0
.,loc:@gradients/read_out_fc2/add_grad/Reshape

:gradients/read_out_fc2/add_grad/tuple/control_dependency_1Identity)gradients/read_out_fc2/add_grad/Reshape_11^gradients/read_out_fc2/add_grad/tuple/group_deps*
_output_shapes
:
*
T0*<
_class2
0.loc:@gradients/read_out_fc2/add_grad/Reshape_1
т
)gradients/read_out_fc2/MatMul_grad/MatMulMatMul8gradients/read_out_fc2/add_grad/tuple/control_dependencyread_out_fc2/Variable/read*(
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
Я
+gradients/read_out_fc2/MatMul_grad/MatMul_1MatMulfc_layer1/Relu8gradients/read_out_fc2/add_grad/tuple/control_dependency*
_output_shapes
:	
*
transpose_a(*
transpose_b( *
T0

3gradients/read_out_fc2/MatMul_grad/tuple/group_depsNoOp*^gradients/read_out_fc2/MatMul_grad/MatMul,^gradients/read_out_fc2/MatMul_grad/MatMul_1

;gradients/read_out_fc2/MatMul_grad/tuple/control_dependencyIdentity)gradients/read_out_fc2/MatMul_grad/MatMul4^gradients/read_out_fc2/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*<
_class2
0.loc:@gradients/read_out_fc2/MatMul_grad/MatMul

=gradients/read_out_fc2/MatMul_grad/tuple/control_dependency_1Identity+gradients/read_out_fc2/MatMul_grad/MatMul_14^gradients/read_out_fc2/MatMul_grad/tuple/group_deps*
_output_shapes
:	
*
T0*>
_class4
20loc:@gradients/read_out_fc2/MatMul_grad/MatMul_1
В
&gradients/fc_layer1/Relu_grad/ReluGradReluGrad;gradients/read_out_fc2/MatMul_grad/tuple/control_dependencyfc_layer1/Relu*(
_output_shapes
:џџџџџџџџџ*
T0
r
"gradients/fc_layer1/add_grad/ShapeShapefc_layer1/MatMul*
_output_shapes
:*
T0*
out_type0
o
$gradients/fc_layer1/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
в
2gradients/fc_layer1/add_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/fc_layer1/add_grad/Shape$gradients/fc_layer1/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
У
 gradients/fc_layer1/add_grad/SumSum&gradients/fc_layer1/Relu_grad/ReluGrad2gradients/fc_layer1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ж
$gradients/fc_layer1/add_grad/ReshapeReshape gradients/fc_layer1/add_grad/Sum"gradients/fc_layer1/add_grad/Shape*(
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0
Ч
"gradients/fc_layer1/add_grad/Sum_1Sum&gradients/fc_layer1/Relu_grad/ReluGrad4gradients/fc_layer1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Џ
&gradients/fc_layer1/add_grad/Reshape_1Reshape"gradients/fc_layer1/add_grad/Sum_1$gradients/fc_layer1/add_grad/Shape_1*
_output_shapes	
:*
T0*
Tshape0

-gradients/fc_layer1/add_grad/tuple/group_depsNoOp%^gradients/fc_layer1/add_grad/Reshape'^gradients/fc_layer1/add_grad/Reshape_1

5gradients/fc_layer1/add_grad/tuple/control_dependencyIdentity$gradients/fc_layer1/add_grad/Reshape.^gradients/fc_layer1/add_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџ*
T0*7
_class-
+)loc:@gradients/fc_layer1/add_grad/Reshape
ќ
7gradients/fc_layer1/add_grad/tuple/control_dependency_1Identity&gradients/fc_layer1/add_grad/Reshape_1.^gradients/fc_layer1/add_grad/tuple/group_deps*
_output_shapes	
:*
T0*9
_class/
-+loc:@gradients/fc_layer1/add_grad/Reshape_1
й
&gradients/fc_layer1/MatMul_grad/MatMulMatMul5gradients/fc_layer1/add_grad/tuple/control_dependencyfc_layer1/Variable/read*
T0*(
_output_shapes
:џџџџџџџџџР*
transpose_a( *
transpose_b(
Э
(gradients/fc_layer1/MatMul_grad/MatMul_1MatMulfc_layer1/Reshape5gradients/fc_layer1/add_grad/tuple/control_dependency* 
_output_shapes
:
Р*
transpose_a(*
transpose_b( *
T0

0gradients/fc_layer1/MatMul_grad/tuple/group_depsNoOp'^gradients/fc_layer1/MatMul_grad/MatMul)^gradients/fc_layer1/MatMul_grad/MatMul_1

8gradients/fc_layer1/MatMul_grad/tuple/control_dependencyIdentity&gradients/fc_layer1/MatMul_grad/MatMul1^gradients/fc_layer1/MatMul_grad/tuple/group_deps*(
_output_shapes
:џџџџџџџџџР*
T0*9
_class/
-+loc:@gradients/fc_layer1/MatMul_grad/MatMul

:gradients/fc_layer1/MatMul_grad/tuple/control_dependency_1Identity(gradients/fc_layer1/MatMul_grad/MatMul_11^gradients/fc_layer1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/fc_layer1/MatMul_grad/MatMul_1* 
_output_shapes
:
Р
u
&gradients/fc_layer1/Reshape_grad/ShapeShapelayer_2/MaxPool*
_output_shapes
:*
T0*
out_type0
н
(gradients/fc_layer1/Reshape_grad/ReshapeReshape8gradients/fc_layer1/MatMul_grad/tuple/control_dependency&gradients/fc_layer1/Reshape_grad/Shape*/
_output_shapes
:џџџџџџџџџ@*
T0*
Tshape0

*gradients/layer_2/MaxPool_grad/MaxPoolGradMaxPoolGradlayer_2/Relulayer_2/MaxPool(gradients/fc_layer1/Reshape_grad/Reshape*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ@*
T0*
data_formatNHWC*
strides

Є
$gradients/layer_2/Relu_grad/ReluGradReluGrad*gradients/layer_2/MaxPool_grad/MaxPoolGradlayer_2/Relu*
T0*/
_output_shapes
:џџџџџџџџџ@
n
 gradients/layer_2/add_grad/ShapeShapelayer_2/Conv2D*
T0*
out_type0*
_output_shapes
:
l
"gradients/layer_2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:@
Ь
0gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/layer_2/add_grad/Shape"gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
gradients/layer_2/add_grad/SumSum$gradients/layer_2/Relu_grad/ReluGrad0gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
З
"gradients/layer_2/add_grad/ReshapeReshapegradients/layer_2/add_grad/Sum gradients/layer_2/add_grad/Shape*/
_output_shapes
:џџџџџџџџџ@*
T0*
Tshape0
С
 gradients/layer_2/add_grad/Sum_1Sum$gradients/layer_2/Relu_grad/ReluGrad2gradients/layer_2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ј
$gradients/layer_2/add_grad/Reshape_1Reshape gradients/layer_2/add_grad/Sum_1"gradients/layer_2/add_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0

+gradients/layer_2/add_grad/tuple/group_depsNoOp#^gradients/layer_2/add_grad/Reshape%^gradients/layer_2/add_grad/Reshape_1

3gradients/layer_2/add_grad/tuple/control_dependencyIdentity"gradients/layer_2/add_grad/Reshape,^gradients/layer_2/add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/layer_2/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџ@
ѓ
5gradients/layer_2/add_grad/tuple/control_dependency_1Identity$gradients/layer_2/add_grad/Reshape_1,^gradients/layer_2/add_grad/tuple/group_deps*
_output_shapes
:@*
T0*7
_class-
+)loc:@gradients/layer_2/add_grad/Reshape_1

$gradients/layer_2/Conv2D_grad/ShapeNShapeNlayer_1/MaxPoollayer_2/Variable/read*
T0*
out_type0*
N* 
_output_shapes
::
|
#gradients/layer_2/Conv2D_grad/ConstConst*%
valueB"          @   *
dtype0*
_output_shapes
:
§
1gradients/layer_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/layer_2/Conv2D_grad/ShapeNlayer_2/Variable/read3gradients/layer_2/add_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
д
2gradients/layer_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterlayer_1/MaxPool#gradients/layer_2/Conv2D_grad/Const3gradients/layer_2/add_grad/tuple/control_dependency*
paddingSAME*&
_output_shapes
: @*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(

.gradients/layer_2/Conv2D_grad/tuple/group_depsNoOp3^gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter2^gradients/layer_2/Conv2D_grad/Conv2DBackpropInput
І
6gradients/layer_2/Conv2D_grad/tuple/control_dependencyIdentity1gradients/layer_2/Conv2D_grad/Conv2DBackpropInput/^gradients/layer_2/Conv2D_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/layer_2/Conv2D_grad/Conv2DBackpropInput*/
_output_shapes
:џџџџџџџџџ 
Ё
8gradients/layer_2/Conv2D_grad/tuple/control_dependency_1Identity2gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter/^gradients/layer_2/Conv2D_grad/tuple/group_deps*&
_output_shapes
: @*
T0*E
_class;
97loc:@gradients/layer_2/Conv2D_grad/Conv2DBackpropFilter

*gradients/layer_1/MaxPool_grad/MaxPoolGradMaxPoolGradlayer_1/Relulayer_1/MaxPool6gradients/layer_2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*/
_output_shapes
:џџџџџџџџџ *
T0*
data_formatNHWC*
strides

Є
$gradients/layer_1/Relu_grad/ReluGradReluGrad*gradients/layer_1/MaxPool_grad/MaxPoolGradlayer_1/Relu*/
_output_shapes
:џџџџџџџџџ *
T0
n
 gradients/layer_1/add_grad/ShapeShapelayer_1/Conv2D*
T0*
out_type0*
_output_shapes
:
l
"gradients/layer_1/add_grad/Shape_1Const*
valueB: *
dtype0*
_output_shapes
:
Ь
0gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/layer_1/add_grad/Shape"gradients/layer_1/add_grad/Shape_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Н
gradients/layer_1/add_grad/SumSum$gradients/layer_1/Relu_grad/ReluGrad0gradients/layer_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
З
"gradients/layer_1/add_grad/ReshapeReshapegradients/layer_1/add_grad/Sum gradients/layer_1/add_grad/Shape*
T0*
Tshape0*/
_output_shapes
:џџџџџџџџџ 
С
 gradients/layer_1/add_grad/Sum_1Sum$gradients/layer_1/Relu_grad/ReluGrad2gradients/layer_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ј
$gradients/layer_1/add_grad/Reshape_1Reshape gradients/layer_1/add_grad/Sum_1"gradients/layer_1/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0

+gradients/layer_1/add_grad/tuple/group_depsNoOp#^gradients/layer_1/add_grad/Reshape%^gradients/layer_1/add_grad/Reshape_1

3gradients/layer_1/add_grad/tuple/control_dependencyIdentity"gradients/layer_1/add_grad/Reshape,^gradients/layer_1/add_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/layer_1/add_grad/Reshape*/
_output_shapes
:џџџџџџџџџ 
ѓ
5gradients/layer_1/add_grad/tuple/control_dependency_1Identity$gradients/layer_1/add_grad/Reshape_1,^gradients/layer_1/add_grad/tuple/group_deps*
_output_shapes
: *
T0*7
_class-
+)loc:@gradients/layer_1/add_grad/Reshape_1

$gradients/layer_1/Conv2D_grad/ShapeNShapeNIteratorGetNextlayer_1/Variable/read*
T0*
out_type0*
N* 
_output_shapes
::
|
#gradients/layer_1/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
§
1gradients/layer_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput$gradients/layer_1/Conv2D_grad/ShapeNlayer_1/Variable/read3gradients/layer_1/add_grad/tuple/control_dependency*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
д
2gradients/layer_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterIteratorGetNext#gradients/layer_1/Conv2D_grad/Const3gradients/layer_1/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*&
_output_shapes
: 

.gradients/layer_1/Conv2D_grad/tuple/group_depsNoOp3^gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter2^gradients/layer_1/Conv2D_grad/Conv2DBackpropInput
І
6gradients/layer_1/Conv2D_grad/tuple/control_dependencyIdentity1gradients/layer_1/Conv2D_grad/Conv2DBackpropInput/^gradients/layer_1/Conv2D_grad/tuple/group_deps*/
_output_shapes
:џџџџџџџџџ*
T0*D
_class:
86loc:@gradients/layer_1/Conv2D_grad/Conv2DBackpropInput
Ё
8gradients/layer_1/Conv2D_grad/tuple/control_dependency_1Identity2gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter/^gradients/layer_1/Conv2D_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/layer_1/Conv2D_grad/Conv2DBackpropFilter*&
_output_shapes
: 
b
GradientDescent/learning_rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Ј
<GradientDescent/update_layer_1/Variable/ApplyGradientDescentApplyGradientDescentlayer_1/VariableGradientDescent/learning_rate8gradients/layer_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_1/Variable*&
_output_shapes
: 

>GradientDescent/update_layer_1/Variable_1/ApplyGradientDescentApplyGradientDescentlayer_1/Variable_1GradientDescent/learning_rate5gradients/layer_1/add_grad/tuple/control_dependency_1*
_output_shapes
: *
use_locking( *
T0*%
_class
loc:@layer_1/Variable_1
Ј
<GradientDescent/update_layer_2/Variable/ApplyGradientDescentApplyGradientDescentlayer_2/VariableGradientDescent/learning_rate8gradients/layer_2/Conv2D_grad/tuple/control_dependency_1*&
_output_shapes
: @*
use_locking( *
T0*#
_class
loc:@layer_2/Variable

>GradientDescent/update_layer_2/Variable_1/ApplyGradientDescentApplyGradientDescentlayer_2/Variable_1GradientDescent/learning_rate5gradients/layer_2/add_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*%
_class
loc:@layer_2/Variable_1
Њ
>GradientDescent/update_fc_layer1/Variable/ApplyGradientDescentApplyGradientDescentfc_layer1/VariableGradientDescent/learning_rate:gradients/fc_layer1/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
Р*
use_locking( *
T0*%
_class
loc:@fc_layer1/Variable
Ј
@GradientDescent/update_fc_layer1/Variable_1/ApplyGradientDescentApplyGradientDescentfc_layer1/Variable_1GradientDescent/learning_rate7gradients/fc_layer1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@fc_layer1/Variable_1*
_output_shapes	
:
Е
AGradientDescent/update_read_out_fc2/Variable/ApplyGradientDescentApplyGradientDescentread_out_fc2/VariableGradientDescent/learning_rate=gradients/read_out_fc2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@read_out_fc2/Variable*
_output_shapes
:	

Г
CGradientDescent/update_read_out_fc2/Variable_1/ApplyGradientDescentApplyGradientDescentread_out_fc2/Variable_1GradientDescent/learning_rate:gradients/read_out_fc2/add_grad/tuple/control_dependency_1*
use_locking( *
T0**
_class 
loc:@read_out_fc2/Variable_1*
_output_shapes
:

Ќ
GradientDescent/updateNoOp?^GradientDescent/update_fc_layer1/Variable/ApplyGradientDescentA^GradientDescent/update_fc_layer1/Variable_1/ApplyGradientDescent=^GradientDescent/update_layer_1/Variable/ApplyGradientDescent?^GradientDescent/update_layer_1/Variable_1/ApplyGradientDescent=^GradientDescent/update_layer_2/Variable/ApplyGradientDescent?^GradientDescent/update_layer_2/Variable_1/ApplyGradientDescentB^GradientDescent/update_read_out_fc2/Variable/ApplyGradientDescentD^GradientDescent/update_read_out_fc2/Variable_1/ApplyGradientDescent

GradientDescent/valueConst^GradientDescent/update*
dtype0	*
_output_shapes
: *
_class
loc:@global_step*
value	B	 R

GradientDescent	AssignAddglobal_stepGradientDescent/value*
use_locking( *
T0	*
_class
loc:@global_step*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 

initNoOp^fc_layer1/Variable/Assign^fc_layer1/Variable_1/Assign^global_step/Assign^layer_1/Variable/Assign^layer_1/Variable_1/Assign^layer_2/Variable/Assign^layer_2/Variable_1/Assign^read_out_fc2/Variable/Assign^read_out_fc2/Variable_1/Assign

init_1NoOp
"

group_depsNoOp^init^init_1

4report_uninitialized_variables/IsVariableInitializedIsVariableInitializedglobal_step*
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
Ћ
6report_uninitialized_variables/IsVariableInitialized_1IsVariableInitializedlayer_1/Variable*#
_class
loc:@layer_1/Variable*
dtype0*
_output_shapes
: 
Џ
6report_uninitialized_variables/IsVariableInitialized_2IsVariableInitializedlayer_1/Variable_1*%
_class
loc:@layer_1/Variable_1*
dtype0*
_output_shapes
: 
Ћ
6report_uninitialized_variables/IsVariableInitialized_3IsVariableInitializedlayer_2/Variable*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/Variable
Џ
6report_uninitialized_variables/IsVariableInitialized_4IsVariableInitializedlayer_2/Variable_1*%
_class
loc:@layer_2/Variable_1*
dtype0*
_output_shapes
: 
Џ
6report_uninitialized_variables/IsVariableInitialized_5IsVariableInitializedfc_layer1/Variable*
dtype0*
_output_shapes
: *%
_class
loc:@fc_layer1/Variable
Г
6report_uninitialized_variables/IsVariableInitialized_6IsVariableInitializedfc_layer1/Variable_1*'
_class
loc:@fc_layer1/Variable_1*
dtype0*
_output_shapes
: 
Е
6report_uninitialized_variables/IsVariableInitialized_7IsVariableInitializedread_out_fc2/Variable*
dtype0*
_output_shapes
: *(
_class
loc:@read_out_fc2/Variable
Й
6report_uninitialized_variables/IsVariableInitialized_8IsVariableInitializedread_out_fc2/Variable_1*
dtype0*
_output_shapes
: **
_class 
loc:@read_out_fc2/Variable_1
Ѓ
6report_uninitialized_variables/IsVariableInitialized_9IsVariableInitializedacc_op/total*
dtype0*
_output_shapes
: *
_class
loc:@acc_op/total
Є
7report_uninitialized_variables/IsVariableInitialized_10IsVariableInitializedacc_op/count*
dtype0*
_output_shapes
: *
_class
loc:@acc_op/count
м
$report_uninitialized_variables/stackPack4report_uninitialized_variables/IsVariableInitialized6report_uninitialized_variables/IsVariableInitialized_16report_uninitialized_variables/IsVariableInitialized_26report_uninitialized_variables/IsVariableInitialized_36report_uninitialized_variables/IsVariableInitialized_46report_uninitialized_variables/IsVariableInitialized_56report_uninitialized_variables/IsVariableInitialized_66report_uninitialized_variables/IsVariableInitialized_76report_uninitialized_variables/IsVariableInitialized_86report_uninitialized_variables/IsVariableInitialized_97report_uninitialized_variables/IsVariableInitialized_10"/device:CPU:0*
T0
*

axis *
N*
_output_shapes
:

)report_uninitialized_variables/LogicalNot
LogicalNot$report_uninitialized_variables/stack"/device:CPU:0*
_output_shapes
:
Ь
$report_uninitialized_variables/ConstConst"/device:CPU:0*
dtype0*
_output_shapes
:*ф
valueкBзBglobal_stepBlayer_1/VariableBlayer_1/Variable_1Blayer_2/VariableBlayer_2/Variable_1Bfc_layer1/VariableBfc_layer1/Variable_1Bread_out_fc2/VariableBread_out_fc2/Variable_1Bacc_op/totalBacc_op/count

1report_uninitialized_variables/boolean_mask/ShapeConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

?report_uninitialized_variables/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
ш
9report_uninitialized_variables/boolean_mask/strided_sliceStridedSlice1report_uninitialized_variables/boolean_mask/Shape?report_uninitialized_variables/boolean_mask/strided_slice/stackAreport_uninitialized_variables/boolean_mask/strided_slice/stack_1Areport_uninitialized_variables/boolean_mask/strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0

Breport_uninitialized_variables/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

0report_uninitialized_variables/boolean_mask/ProdProd9report_uninitialized_variables/boolean_mask/strided_sliceBreport_uninitialized_variables/boolean_mask/Prod/reduction_indices"/device:CPU:0*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

3report_uninitialized_variables/boolean_mask/Shape_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
№
;report_uninitialized_variables/boolean_mask/strided_slice_1StridedSlice3report_uninitialized_variables/boolean_mask/Shape_1Areport_uninitialized_variables/boolean_mask/strided_slice_1/stackCreport_uninitialized_variables/boolean_mask/strided_slice_1/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0

3report_uninitialized_variables/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
№
;report_uninitialized_variables/boolean_mask/strided_slice_2StridedSlice3report_uninitialized_variables/boolean_mask/Shape_2Areport_uninitialized_variables/boolean_mask/strided_slice_2/stackCreport_uninitialized_variables/boolean_mask/strided_slice_2/stack_1Creport_uninitialized_variables/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
end_mask*
_output_shapes
: *
T0*
Index0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask 
О
;report_uninitialized_variables/boolean_mask/concat/values_1Pack0report_uninitialized_variables/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

7report_uninitialized_variables/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
ї
2report_uninitialized_variables/boolean_mask/concatConcatV2;report_uninitialized_variables/boolean_mask/strided_slice_1;report_uninitialized_variables/boolean_mask/concat/values_1;report_uninitialized_variables/boolean_mask/strided_slice_27report_uninitialized_variables/boolean_mask/concat/axis"/device:CPU:0*
N*
_output_shapes
:*

Tidx0*
T0
к
3report_uninitialized_variables/boolean_mask/ReshapeReshape$report_uninitialized_variables/Const2report_uninitialized_variables/boolean_mask/concat"/device:CPU:0*
_output_shapes
:*
T0*
Tshape0

;report_uninitialized_variables/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
ъ
5report_uninitialized_variables/boolean_mask/Reshape_1Reshape)report_uninitialized_variables/LogicalNot;report_uninitialized_variables/boolean_mask/Reshape_1/shape"/device:CPU:0*
_output_shapes
:*
T0
*
Tshape0
В
1report_uninitialized_variables/boolean_mask/WhereWhere5report_uninitialized_variables/boolean_mask/Reshape_1"/device:CPU:0*'
_output_shapes
:џџџџџџџџџ*
T0

Х
3report_uninitialized_variables/boolean_mask/SqueezeSqueeze1report_uninitialized_variables/boolean_mask/Where"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
*
T0	

9report_uninitialized_variables/boolean_mask/GatherV2/axisConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
Х
4report_uninitialized_variables/boolean_mask/GatherV2GatherV23report_uninitialized_variables/boolean_mask/Reshape3report_uninitialized_variables/boolean_mask/Squeeze9report_uninitialized_variables/boolean_mask/GatherV2/axis"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ*
Taxis0*
Tindices0	*
Tparams0
v
$report_uninitialized_resources/ConstConst"/device:CPU:0*
valueB *
dtype0*
_output_shapes
: 
M
concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
О
concatConcatV24report_uninitialized_variables/boolean_mask/GatherV2$report_uninitialized_resources/Constconcat/axis*
N*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
Ё
6report_uninitialized_variables_1/IsVariableInitializedIsVariableInitializedglobal_step*
dtype0	*
_output_shapes
: *
_class
loc:@global_step
­
8report_uninitialized_variables_1/IsVariableInitialized_1IsVariableInitializedlayer_1/Variable*#
_class
loc:@layer_1/Variable*
dtype0*
_output_shapes
: 
Б
8report_uninitialized_variables_1/IsVariableInitialized_2IsVariableInitializedlayer_1/Variable_1*%
_class
loc:@layer_1/Variable_1*
dtype0*
_output_shapes
: 
­
8report_uninitialized_variables_1/IsVariableInitialized_3IsVariableInitializedlayer_2/Variable*
dtype0*
_output_shapes
: *#
_class
loc:@layer_2/Variable
Б
8report_uninitialized_variables_1/IsVariableInitialized_4IsVariableInitializedlayer_2/Variable_1*
dtype0*
_output_shapes
: *%
_class
loc:@layer_2/Variable_1
Б
8report_uninitialized_variables_1/IsVariableInitialized_5IsVariableInitializedfc_layer1/Variable*
dtype0*
_output_shapes
: *%
_class
loc:@fc_layer1/Variable
Е
8report_uninitialized_variables_1/IsVariableInitialized_6IsVariableInitializedfc_layer1/Variable_1*
dtype0*
_output_shapes
: *'
_class
loc:@fc_layer1/Variable_1
З
8report_uninitialized_variables_1/IsVariableInitialized_7IsVariableInitializedread_out_fc2/Variable*(
_class
loc:@read_out_fc2/Variable*
dtype0*
_output_shapes
: 
Л
8report_uninitialized_variables_1/IsVariableInitialized_8IsVariableInitializedread_out_fc2/Variable_1*
dtype0*
_output_shapes
: **
_class 
loc:@read_out_fc2/Variable_1
џ
&report_uninitialized_variables_1/stackPack6report_uninitialized_variables_1/IsVariableInitialized8report_uninitialized_variables_1/IsVariableInitialized_18report_uninitialized_variables_1/IsVariableInitialized_28report_uninitialized_variables_1/IsVariableInitialized_38report_uninitialized_variables_1/IsVariableInitialized_48report_uninitialized_variables_1/IsVariableInitialized_58report_uninitialized_variables_1/IsVariableInitialized_68report_uninitialized_variables_1/IsVariableInitialized_78report_uninitialized_variables_1/IsVariableInitialized_8"/device:CPU:0*
T0
*

axis *
N	*
_output_shapes
:	

+report_uninitialized_variables_1/LogicalNot
LogicalNot&report_uninitialized_variables_1/stack"/device:CPU:0*
_output_shapes
:	
В
&report_uninitialized_variables_1/ConstConst"/device:CPU:0*Ш
valueОBЛ	Bglobal_stepBlayer_1/VariableBlayer_1/Variable_1Blayer_2/VariableBlayer_2/Variable_1Bfc_layer1/VariableBfc_layer1/Variable_1Bread_out_fc2/VariableBread_out_fc2/Variable_1*
dtype0*
_output_shapes
:	

3report_uninitialized_variables_1/boolean_mask/ShapeConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:	

Areport_uninitialized_variables_1/boolean_mask/strided_slice/stackConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2Const"/device:CPU:0*
valueB:*
dtype0*
_output_shapes
:
ђ
;report_uninitialized_variables_1/boolean_mask/strided_sliceStridedSlice3report_uninitialized_variables_1/boolean_mask/ShapeAreport_uninitialized_variables_1/boolean_mask/strided_slice/stackCreport_uninitialized_variables_1/boolean_mask/strided_slice/stack_1Creport_uninitialized_variables_1/boolean_mask/strided_slice/stack_2"/device:CPU:0*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0

Dreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indicesConst"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

2report_uninitialized_variables_1/boolean_mask/ProdProd;report_uninitialized_variables_1/boolean_mask/strided_sliceDreport_uninitialized_variables_1/boolean_mask/Prod/reduction_indices"/device:CPU:0*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0

5report_uninitialized_variables_1/boolean_mask/Shape_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:	

Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB: 

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_1StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_1Creport_uninitialized_variables_1/boolean_mask/strided_slice_1/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_1/stack_2"/device:CPU:0*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask 

5report_uninitialized_variables_1/boolean_mask/Shape_2Const"/device:CPU:0*
valueB:	*
dtype0*
_output_shapes
:

Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Const"/device:CPU:0*
valueB: *
dtype0*
_output_shapes
:

Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2Const"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB:
њ
=report_uninitialized_variables_1/boolean_mask/strided_slice_2StridedSlice5report_uninitialized_variables_1/boolean_mask/Shape_2Creport_uninitialized_variables_1/boolean_mask/strided_slice_2/stackEreport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_1Ereport_uninitialized_variables_1/boolean_mask/strided_slice_2/stack_2"/device:CPU:0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
: *
Index0*
T0
Т
=report_uninitialized_variables_1/boolean_mask/concat/values_1Pack2report_uninitialized_variables_1/boolean_mask/Prod"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

9report_uninitialized_variables_1/boolean_mask/concat/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

4report_uninitialized_variables_1/boolean_mask/concatConcatV2=report_uninitialized_variables_1/boolean_mask/strided_slice_1=report_uninitialized_variables_1/boolean_mask/concat/values_1=report_uninitialized_variables_1/boolean_mask/strided_slice_29report_uninitialized_variables_1/boolean_mask/concat/axis"/device:CPU:0*
N*
_output_shapes
:*

Tidx0*
T0
р
5report_uninitialized_variables_1/boolean_mask/ReshapeReshape&report_uninitialized_variables_1/Const4report_uninitialized_variables_1/boolean_mask/concat"/device:CPU:0*
T0*
Tshape0*
_output_shapes
:	

=report_uninitialized_variables_1/boolean_mask/Reshape_1/shapeConst"/device:CPU:0*
valueB:
џџџџџџџџџ*
dtype0*
_output_shapes
:
№
7report_uninitialized_variables_1/boolean_mask/Reshape_1Reshape+report_uninitialized_variables_1/LogicalNot=report_uninitialized_variables_1/boolean_mask/Reshape_1/shape"/device:CPU:0*
_output_shapes
:	*
T0
*
Tshape0
Ж
3report_uninitialized_variables_1/boolean_mask/WhereWhere7report_uninitialized_variables_1/boolean_mask/Reshape_1"/device:CPU:0*
T0
*'
_output_shapes
:џџџџџџџџџ
Щ
5report_uninitialized_variables_1/boolean_mask/SqueezeSqueeze3report_uninitialized_variables_1/boolean_mask/Where"/device:CPU:0*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
*
T0	

;report_uninitialized_variables_1/boolean_mask/GatherV2/axisConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
Э
6report_uninitialized_variables_1/boolean_mask/GatherV2GatherV25report_uninitialized_variables_1/boolean_mask/Reshape5report_uninitialized_variables_1/boolean_mask/Squeeze;report_uninitialized_variables_1/boolean_mask/GatherV2/axis"/device:CPU:0*
Tindices0	*
Tparams0*#
_output_shapes
:џџџџџџџџџ*
Taxis0
:
init_2NoOp^acc_op/count/Assign^acc_op/total/Assign

init_all_tablesNoOp

init_3NoOp
8
group_deps_1NoOp^init_2^init_3^init_all_tables
S
Merge/MergeSummaryMergeSummaryaccuracyloss*
N*
_output_shapes
: 
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_6f190f302adc4398a12d5894a00594e7/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
Є
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:	*Ш
valueОBЛ	Bfc_layer1/VariableBfc_layer1/Variable_1Bglobal_stepBlayer_1/VariableBlayer_1/Variable_1Blayer_2/VariableBlayer_2/Variable_1Bread_out_fc2/VariableBread_out_fc2/Variable_1

save/SaveV2/shape_and_slicesConst"/device:CPU:0*%
valueB	B B B B B B B B B *
dtype0*
_output_shapes
:	
О
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesfc_layer1/Variablefc_layer1/Variable_1global_steplayer_1/Variablelayer_1/Variable_1layer_2/Variablelayer_2/Variable_1read_out_fc2/Variableread_out_fc2/Variable_1"/device:CPU:0*
dtypes
2		
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Ќ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
Ї
save/RestoreV2/tensor_namesConst"/device:CPU:0*Ш
valueОBЛ	Bfc_layer1/VariableBfc_layer1/Variable_1Bglobal_stepBlayer_1/VariableBlayer_1/Variable_1Blayer_2/VariableBlayer_2/Variable_1Bread_out_fc2/VariableBread_out_fc2/Variable_1*
dtype0*
_output_shapes
:	

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:	*%
valueB	B B B B B B B B B 
Ч
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2		
Д
save/AssignAssignfc_layer1/Variablesave/RestoreV2*
validate_shape(* 
_output_shapes
:
Р*
use_locking(*
T0*%
_class
loc:@fc_layer1/Variable
З
save/Assign_1Assignfc_layer1/Variable_1save/RestoreV2:1*
use_locking(*
T0*'
_class
loc:@fc_layer1/Variable_1*
validate_shape(*
_output_shapes	
:
 
save/Assign_2Assignglobal_stepsave/RestoreV2:2*
use_locking(*
T0	*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
К
save/Assign_3Assignlayer_1/Variablesave/RestoreV2:3*
validate_shape(*&
_output_shapes
: *
use_locking(*
T0*#
_class
loc:@layer_1/Variable
В
save/Assign_4Assignlayer_1/Variable_1save/RestoreV2:4*
use_locking(*
T0*%
_class
loc:@layer_1/Variable_1*
validate_shape(*
_output_shapes
: 
К
save/Assign_5Assignlayer_2/Variablesave/RestoreV2:5*
use_locking(*
T0*#
_class
loc:@layer_2/Variable*
validate_shape(*&
_output_shapes
: @
В
save/Assign_6Assignlayer_2/Variable_1save/RestoreV2:6*
use_locking(*
T0*%
_class
loc:@layer_2/Variable_1*
validate_shape(*
_output_shapes
:@
Н
save/Assign_7Assignread_out_fc2/Variablesave/RestoreV2:7*
validate_shape(*
_output_shapes
:	
*
use_locking(*
T0*(
_class
loc:@read_out_fc2/Variable
М
save/Assign_8Assignread_out_fc2/Variable_1save/RestoreV2:8*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0**
_class 
loc:@read_out_fc2/Variable_1
Ј
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8
-
save/restore_allNoOp^save/restore_shardУ+
ю

tf_map_func_6yLLF6rv0aU
arg0
resize_images_squeeze

cond_merge25A wrapper for Defun that facilitates shape inference./
ConstConst*
value	B B/*
dtype02
packedPackarg0*
N*
T0*

axis M
StringSplitStringSplitpacked:output:0Const:output:0*

skip_empty(J
strided_slice/stackConst*
valueB:
ўџџџџџџџџ*
dtype0L
strided_slice/stack_1Const*
dtype0*
valueB:
џџџџџџџџџC
strided_slice/stack_2Const*
dtype0*
valueB:
strided_sliceStridedSliceStringSplit:values:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 4
Equal/yConst*
dtype0*
valueB
 BdogsA
EqualEqualstrided_slice:output:0Equal/y:output:0*
T04
cond/SwitchSwitch	Equal:z:0	Equal:z:0*
T0
=
cond/switch_tIdentitycond/Switch:output_true:0*
T0
>
cond/switch_fIdentitycond/Switch:output_false:0*
T0
,
cond/pred_idIdentity	Equal:z:0*
T0
D

cond/ConstConst^cond/switch_t*
value	B : *
dtype0F
cond/Const_1Const^cond/switch_f*
dtype0*
value	B :Q

cond/MergeMergecond/Const_1:output:0cond/Const:output:0*
T0*
N
ReadFileReadFilearg0Ў

DecodeJpeg
DecodeJpegReadFile:contents:0*
ratio*

dct_method *
channels*
acceptable_fraction%  ?*
fancy_upscaling(*
try_recover_truncated( F
convert_image/CastCastDecodeJpeg:image:0*

DstT0*

SrcT0<
convert_image/yConst*
dtype0*
valueB
 *;O
convert_imageMulconvert_image/Cast:y:0convert_image/y:output:0*
T0F
resize_images/ExpandDims/dimConst*
value	B : *
dtype0u
resize_images/ExpandDims
ExpandDimsconvert_image:z:0%resize_images/ExpandDims/dim:output:0*

Tdim0*
T0G
resize_images/sizeConst*
valueB"      *
dtype0
resize_images/ResizeBilinearResizeBilinear!resize_images/ExpandDims:output:0resize_images/size:output:0*
align_corners( *
T0o
resize_images/SqueezeSqueeze-resize_images/ResizeBilinear:resized_images:0*
squeeze_dims
 *
T0"!

cond_mergecond/Merge:output:0"7
resize_images_squeezeresize_images/Squeeze:output:0
Я
3
_make_dataset_GUdGPfB2wf0
prefetchdatasetn
(TensorSliceDataset/MatchingFiles/patternConst*
dtype0*.
value%B# B/data/dogscats/train/**/*.jpgd
 TensorSliceDataset/MatchingFilesMatchingFiles1TensorSliceDataset/MatchingFiles/pattern:output:0
TensorSliceDatasetTensorSliceDataset,TensorSliceDataset/MatchingFiles:filenames:0*
output_shapes
: *
Toutput_types
2j
$ShuffleDataset/MatchingFiles/patternConst*.
value%B# B/data/dogscats/train/**/*.jpg*
dtype0\
ShuffleDataset/MatchingFilesMatchingFiles-ShuffleDataset/MatchingFiles/pattern:output:0`
ShuffleDataset/ShapeShape(ShuffleDataset/MatchingFiles:filenames:0*
T0*
out_type0	P
"ShuffleDataset/strided_slice/stackConst*
valueB: *
dtype0R
$ShuffleDataset/strided_slice/stack_1Const*
valueB:*
dtype0R
$ShuffleDataset/strided_slice/stack_2Const*
valueB:*
dtype0а
ShuffleDataset/strided_sliceStridedSliceShuffleDataset/Shape:output:0+ShuffleDataset/strided_slice/stack:output:0-ShuffleDataset/strided_slice/stack_1:output:0-ShuffleDataset/strided_slice/stack_2:output:0*
Index0*
T0	*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask B
ShuffleDataset/Maximum/yConst*
value	B	 R*
dtype0	t
ShuffleDataset/MaximumMaximum%ShuffleDataset/strided_slice:output:0!ShuffleDataset/Maximum/y:output:0*
T0	=
ShuffleDataset/seedConst*
dtype0	*
value	B	 R >
ShuffleDataset/seed2Const*
dtype0	*
value	B	 R ф
ShuffleDatasetShuffleDatasetTensorSliceDataset:handle:0ShuffleDataset/Maximum:z:0ShuffleDataset/seed:output:0ShuffleDataset/seed2:output:0*
output_shapes
: *
reshuffle_each_iteration(*
output_types
2N
#ShuffleAndRepeatDataset/buffer_sizeConst*
value
B	 R *
dtype0	H
ShuffleAndRepeatDataset/seed_1Const*
dtype0	*
value	B	 R I
ShuffleAndRepeatDataset/seed2_1Const*
dtype0	*
value	B	 R G
ShuffleAndRepeatDataset/countConst*
value	B	 R
*
dtype0	Ђ
ShuffleAndRepeatDatasetShuffleAndRepeatDatasetShuffleDataset:handle:0,ShuffleAndRepeatDataset/buffer_size:output:0'ShuffleAndRepeatDataset/seed_1:output:0(ShuffleAndRepeatDataset/seed2_1:output:0&ShuffleAndRepeatDataset/count:output:0*
output_shapes
: *
output_types
2Ћ

MapDataset
MapDataset ShuffleAndRepeatDataset:handle:0* 
fR
tf_map_func_6yLLF6rv0aU*
output_types
2*

Targuments
 *#
output_shapes
:: A
BatchDataset/batch_sizeConst*
value	B	 R@*
dtype0	Њ
BatchDatasetBatchDatasetMapDataset:handle:0 BatchDataset/batch_size:output:0*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ*
output_types
2H
PrefetchDataset/buffer_size_1Const*
value
B	 R *
dtype0	И
PrefetchDatasetPrefetchDatasetBatchDataset:handle:0&PrefetchDataset/buffer_size_1:output:0*
output_types
2*=
output_shapes,
*:џџџџџџџџџ:џџџџџџџџџ"+
prefetchdatasetPrefetchDataset:handle:0"<
save/Const:0save/Identity:0save/restore_all (5 @F8"W
ready_for_local_init_op<
:
8report_uninitialized_variables_1/boolean_mask/GatherV2:0"
init_op


group_deps"Р
cond_contextЏЌ

global_step/cond/cond_textglobal_step/cond/pred_id:0global_step/cond/switch_t:0 *Ј
global_step/cond/pred_id:0
global_step/cond/read/Switch:1
global_step/cond/read:0
global_step/cond/switch_t:0
global_step:0/
global_step:0global_step/cond/read/Switch:18
global_step/cond/pred_id:0global_step/cond/pred_id:0:
global_step/cond/switch_t:0global_step/cond/switch_t:0
Є
global_step/cond/cond_text_1global_step/cond/pred_id:0global_step/cond/switch_f:0*Ъ
global_step/Initializer/zeros:0
global_step/cond/Switch_1:0
global_step/cond/Switch_1:1
global_step/cond/pred_id:0
global_step/cond/switch_f:0>
global_step/Initializer/zeros:0global_step/cond/Switch_1:0:
global_step/cond/switch_f:0global_step/cond/switch_f:08
global_step/cond/pred_id:0global_step/cond/pred_id:0"6
metric_variables"
 
acc_op/total:0
acc_op/count:0"г
local_variablesПМ
\
acc_op/total:0acc_op/total/Assignacc_op/total/read:02 acc_op/total/Initializer/zeros:0
\
acc_op/count:0acc_op/count/Assignacc_op/count/read:02 acc_op/count/Initializer/zeros:0"!
local_init_op

group_deps_1"Џ
	variablesЁ
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
b
layer_1/Variable:0layer_1/Variable/Assignlayer_1/Variable/read:02layer_1/truncated_normal:0
]
layer_1/Variable_1:0layer_1/Variable_1/Assignlayer_1/Variable_1/read:02layer_1/Const:0
b
layer_2/Variable:0layer_2/Variable/Assignlayer_2/Variable/read:02layer_2/truncated_normal:0
]
layer_2/Variable_1:0layer_2/Variable_1/Assignlayer_2/Variable_1/read:02layer_2/Const:0
j
fc_layer1/Variable:0fc_layer1/Variable/Assignfc_layer1/Variable/read:02fc_layer1/truncated_normal:0
e
fc_layer1/Variable_1:0fc_layer1/Variable_1/Assignfc_layer1/Variable_1/read:02fc_layer1/Const:0
v
read_out_fc2/Variable:0read_out_fc2/Variable/Assignread_out_fc2/Variable/read:02read_out_fc2/truncated_normal:0
q
read_out_fc2/Variable_1:0read_out_fc2/Variable_1/Assignread_out_fc2/Variable_1/read:02read_out_fc2/Const:0"
ready_op


concat:0"J
savers@>
<
save/Const:0save/Identity:0save/restore_all (5 @F8"
losses


Mean:0"2
global_step_read_op_cache

global_step/add:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"
train_op

GradientDescent"&

summary_op

Merge/MergeSummary:0"п
trainable_variablesЧФ
b
layer_1/Variable:0layer_1/Variable/Assignlayer_1/Variable/read:02layer_1/truncated_normal:0
]
layer_1/Variable_1:0layer_1/Variable_1/Assignlayer_1/Variable_1/read:02layer_1/Const:0
b
layer_2/Variable:0layer_2/Variable/Assignlayer_2/Variable/read:02layer_2/truncated_normal:0
]
layer_2/Variable_1:0layer_2/Variable_1/Assignlayer_2/Variable_1/read:02layer_2/Const:0
j
fc_layer1/Variable:0fc_layer1/Variable/Assignfc_layer1/Variable/read:02fc_layer1/truncated_normal:0
e
fc_layer1/Variable_1:0fc_layer1/Variable_1/Assignfc_layer1/Variable_1/read:02fc_layer1/Const:0
v
read_out_fc2/Variable:0read_out_fc2/Variable/Assignread_out_fc2/Variable/read:02read_out_fc2/truncated_normal:0
q
read_out_fc2/Variable_1:0read_out_fc2/Variable_1/Assignread_out_fc2/Variable_1/read:02read_out_fc2/Const:0"#
	summaries


accuracy:0
loss:0Ёџџ<'       ЛсБF	ЯЕў#>ЩжA:./model_dir/model.ckpt7цW       mS+		yћў#>ЩжA:'Z(+       УK	ўў#>ЩжA*

accuracy   =

loss]ђ@WД%       ъМ6ѓ	Г4>ЩжAe*

global_step/secЁЧ?МB+       УK	4>ЩжAe*

accuracy  h>

lossщ!<?Tшит&       sOу 	AYшD>ЩжAЩ*

global_step/secгН?1Oхy,       єЎЬE	ЩiшD>ЩжAЩ*

accuracyЋЊЊ>

losshІ0?ўьі&       sOу 	иBW>ЩжA­*

global_step/secSbЎ?lю,       єЎЬE	ГBW>ЩжA­*

accuracy  И>

lossД$7?лBЕ&       sOу 	Цi>ЩжA*

global_step/secЂгЌ?6<,       єЎЬE	юЦi>ЩжA*

accuracyЩ>

lossЈ91?є?,&       sOу 	h}>ЩжAѕ*

global_step/secшљЂ?Hь,       єЎЬE	h}>ЩжAѕ*

accuracyЋЊж>

lossтФ0?,'&       sOу 	Ё4Г>ЩжAй*

global_step/secqёЎ?уњЎ,       єЎЬE	XГ>ЩжAй*

accuracyЗmл>

lossЦw4?7i#Г&       sOу 	Cy[І>ЩжAН*

global_step/sec<?Њ3~,       єЎЬE	#[І>ЩжAН*

accuracy  ф>

lossx20?C85Ш(       џpJ	m42К>ЩжA:./model_dir/model.ckpt|u­&       sOу 	 ЋО>ЩжAЁ*

global_step/secм ?Єъ<Ш,       єЎЬE	{ ЋО>ЩжAЁ*

accuracyrч>

loss6І4? 	Ѕ&       sOу 	Ю>ЩжA*

global_step/secГЩ?.ќя,       єЎЬE	+Ю>ЩжA*

accuracyffц>

lossHG2?ыљF&       sOу 	l6с>ЩжAщ*

global_step/secSЋ?КДЌ,       єЎЬE	Ж.6с>ЩжAщ*

accuracyЃю>

lossЩ:0?h<J5