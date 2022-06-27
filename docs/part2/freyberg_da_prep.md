---
layout: default
title: PEST++DA - Getting Ready
parent: Decision Support Modelling with pyEMU and PEST++
nav_order: 10
math: mathjax3
---

# Prepare for sequential data assimilation

Sequential state-parameter estimation is a whole new beast for the PEST world.  Every other tool in PEST and PEST++ operate on the concept of "batch" estimation, where the model is run forward for the full simulation period and PEST(++) simply calls the model and reads the results.  In sequential estimation, PESTPP-DA takes control of the advancing of simulation time.  This opens up some powerful new analyses but requires us to heavily modify the PEST interface and model itself.  This horrible notebook does that...

### 1. The modified Freyberg PEST dataset

The modified Freyberg model is introduced in another tutorial notebook (see "freyberg intro to model"). The current notebook picks up following the "freyberg psfrom pest setup" notebook, in which a high-dimensional PEST dataset was constructed using `pyemu.PstFrom`. You may also wish to go through the "intro to pyemu" notebook beforehand.

The next couple of cells load necessary dependencies and call a convenience function to prepare the PEST dataset folder for you. This is the same dataset that was constructed during the "freyberg pstfrom pest setup" tutorial. Simply press `shift+enter` to run the cells.


```python
import os
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;


import sys
sys.path.append(os.path.join("..", "..", "dependencies"))
import pyemu
import flopy
sys.path.append("..")
# import pre-prepared convenience functions
import herebedragons as hbd



```


```python
# specify the temporary working folder
t_d = os.path.join('freyberg6_da_template')

org_t_d = os.path.join("..","part2_2_obs_and_weights","freyberg6_template")
if not os.path.exists(org_t_d):
    raise Exception("you need to run the '/part2_2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")

if os.path.exists(t_d):
    shutil.rmtree(t_d)
shutil.copytree(org_t_d,t_d)
```




    'freyberg6_da_template'



There are several modifications we need to make to both the model and pest interface in order to go from batch estimation to sequential estimation.  First, we need to make the model a single stress period model - PESTPP-DA will take control of the advancement of simulation time...


```python
with open(os.path.join(t_d,"freyberg6.tdis"),'w') as f:
    f.write("# new tdis written hastily at {0}\n]\n".format(datetime.now()))
    f.write("BEGIN options\n  TIME_UNITS days\nEND options\n\n")
    f.write("BEGIN dimensions\n  NPER 1\nEND dimensions\n\n")
    f.write("BEGIN perioddata\n  1.0  1 1.0\nEND perioddata\n\n")

          
```

Now, just make sure we havent done something dumb (er than usual):


```python
pyemu.os_utils.run("mf6",cwd=t_d)
```

# Now for the hard part

First, let's assign cycle numbers to the time-varying parameters and their template files.  The "cycle" concept is core to squential estimation with PESTPP-DA.  A cycle can be thought of as a unit of simulation time that we are interested in. In the PEST interface, a cycle defines a set of parameters and observations, so you can think of a cycle as a "sub-problem" in the PEST since - PESTPP-DA creates this subproblem under the hood for us. For a given cycle, we will "assimilate" all non-zero weighted obsevations in that cycle using the adjustable parameters and states in that cycle.  If a parameter/observation (and associated input/outputs files) are assigned a cycle value of -1, that means it applies to all cycles. 


```python
pst_path = os.path.join(t_d, 'freyberg_mf6.pst')
pst = pyemu.Pst(pst_path)
if "observed" not in pst.observation_data.columns:
    raise Exception("you need to run the '/part2_obs_and_weights/freyberg_obs_and_weights.ipynb' notebook")
```


```python
df = pst.model_input_data
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pest_file</th>
      <th>model_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>npfklayer1gr_inst0_grid.csv.tpl</td>
      <td>mult\npfklayer1gr_inst0_grid.csv</td>
    </tr>
    <tr>
      <th>1</th>
      <td>npfklayer1pp_inst0pp.dat.tpl</td>
      <td>npfklayer1pp_inst0pp.dat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>npfklayer1cn_inst0_constant.csv.tpl</td>
      <td>mult\npfklayer1cn_inst0_constant.csv</td>
    </tr>
    <tr>
      <th>3</th>
      <td>npfklayer2gr_inst0_grid.csv.tpl</td>
      <td>mult\npfklayer2gr_inst0_grid.csv</td>
    </tr>
    <tr>
      <th>4</th>
      <td>npfklayer2pp_inst0pp.dat.tpl</td>
      <td>npfklayer2pp_inst0pp.dat</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>190</th>
      <td>sfrgr_inst23_grid.csv.tpl</td>
      <td>mult\sfrgr_inst23_grid.csv</td>
    </tr>
    <tr>
      <th>191</th>
      <td>sfrgr_inst24_grid.csv.tpl</td>
      <td>mult\sfrgr_inst24_grid.csv</td>
    </tr>
    <tr>
      <th>192</th>
      <td>freyberg6.ic_strt_layer1.txt.tpl</td>
      <td>org\freyberg6.ic_strt_layer1.txt</td>
    </tr>
    <tr>
      <th>193</th>
      <td>freyberg6.ic_strt_layer2.txt.tpl</td>
      <td>org\freyberg6.ic_strt_layer2.txt</td>
    </tr>
    <tr>
      <th>194</th>
      <td>freyberg6.ic_strt_layer3.txt.tpl</td>
      <td>org\freyberg6.ic_strt_layer3.txt</td>
    </tr>
  </tbody>
</table>
<p>195 rows × 2 columns</p>
</div>




```python
df.loc[:,"cycle"] = -1
```

Here we want to assign the template file info associated with time-varying SFR parameters to the appropriate cycle - this includes resetting the actual model-input filename since we only have only stress period in the model now


```python
sfrdf = df.loc[df.pest_file.apply(lambda x: "sfr" in x and "cond" not in x),:]
sfrdf.loc[:,"inst"] = sfrdf.pest_file.apply(lambda x: int(x.split("inst")[1].split("_")[0]))
sfrdf.loc[:,"model_file"] = sfrdf.model_file.iloc[0]
sfrdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pest_file</th>
      <th>model_file</th>
      <th>cycle</th>
      <th>inst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>167</th>
      <td>sfrgr_inst0_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>-1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>168</th>
      <td>sfrgr_inst1_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>169</th>
      <td>sfrgr_inst2_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>-1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>170</th>
      <td>sfrgr_inst3_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>-1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>171</th>
      <td>sfrgr_inst4_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>-1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[sfrdf.index,"cycle"] = sfrdf.inst.values
df.loc[sfrdf.index,"model_file"] = sfrdf.model_file.values

df.loc[sfrdf.index,:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pest_file</th>
      <th>model_file</th>
      <th>cycle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>167</th>
      <td>sfrgr_inst0_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>0</td>
    </tr>
    <tr>
      <th>168</th>
      <td>sfrgr_inst1_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>1</td>
    </tr>
    <tr>
      <th>169</th>
      <td>sfrgr_inst2_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>2</td>
    </tr>
    <tr>
      <th>170</th>
      <td>sfrgr_inst3_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>3</td>
    </tr>
    <tr>
      <th>171</th>
      <td>sfrgr_inst4_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>4</td>
    </tr>
    <tr>
      <th>172</th>
      <td>sfrgr_inst5_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>5</td>
    </tr>
    <tr>
      <th>173</th>
      <td>sfrgr_inst6_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>6</td>
    </tr>
    <tr>
      <th>174</th>
      <td>sfrgr_inst7_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>7</td>
    </tr>
    <tr>
      <th>175</th>
      <td>sfrgr_inst8_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>8</td>
    </tr>
    <tr>
      <th>176</th>
      <td>sfrgr_inst9_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>9</td>
    </tr>
    <tr>
      <th>177</th>
      <td>sfrgr_inst10_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>10</td>
    </tr>
    <tr>
      <th>178</th>
      <td>sfrgr_inst11_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>11</td>
    </tr>
    <tr>
      <th>179</th>
      <td>sfrgr_inst12_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>12</td>
    </tr>
    <tr>
      <th>180</th>
      <td>sfrgr_inst13_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>13</td>
    </tr>
    <tr>
      <th>181</th>
      <td>sfrgr_inst14_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>14</td>
    </tr>
    <tr>
      <th>182</th>
      <td>sfrgr_inst15_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>15</td>
    </tr>
    <tr>
      <th>183</th>
      <td>sfrgr_inst16_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>16</td>
    </tr>
    <tr>
      <th>184</th>
      <td>sfrgr_inst17_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>17</td>
    </tr>
    <tr>
      <th>185</th>
      <td>sfrgr_inst18_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>18</td>
    </tr>
    <tr>
      <th>186</th>
      <td>sfrgr_inst19_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>19</td>
    </tr>
    <tr>
      <th>187</th>
      <td>sfrgr_inst20_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>20</td>
    </tr>
    <tr>
      <th>188</th>
      <td>sfrgr_inst21_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>21</td>
    </tr>
    <tr>
      <th>189</th>
      <td>sfrgr_inst22_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>22</td>
    </tr>
    <tr>
      <th>190</th>
      <td>sfrgr_inst23_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>23</td>
    </tr>
    <tr>
      <th>191</th>
      <td>sfrgr_inst24_grid.csv.tpl</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



And the same for the template files associated with the WEL package time-varying parameters 


```python
weldf = df.loc[df.pest_file.str.contains('wel'),:]
df.loc[weldf.index,"cycle"] = weldf.pest_file.apply(lambda x: int(x.split("inst")[1].split("_")[0]))
grdf = weldf.loc[weldf.pest_file.str.contains("welgrd"),:]
df.loc[grdf.index,"model_file"] = grdf.model_file.iloc[0]
cndf = weldf.loc[weldf.pest_file.str.contains("welcst"),:]
df.loc[cndf.index,"model_file"] = cndf.model_file.iloc[0]

df.loc[weldf.index,:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pest_file</th>
      <th>model_file</th>
      <th>cycle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>115</th>
      <td>welcst_inst0_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>welgrd_inst0_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>welcst_inst1_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>1</td>
    </tr>
    <tr>
      <th>118</th>
      <td>welgrd_inst1_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>1</td>
    </tr>
    <tr>
      <th>119</th>
      <td>welcst_inst2_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>2</td>
    </tr>
    <tr>
      <th>120</th>
      <td>welgrd_inst2_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>2</td>
    </tr>
    <tr>
      <th>121</th>
      <td>welcst_inst3_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>3</td>
    </tr>
    <tr>
      <th>122</th>
      <td>welgrd_inst3_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>3</td>
    </tr>
    <tr>
      <th>123</th>
      <td>welcst_inst4_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>4</td>
    </tr>
    <tr>
      <th>124</th>
      <td>welgrd_inst4_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>4</td>
    </tr>
    <tr>
      <th>125</th>
      <td>welcst_inst5_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>5</td>
    </tr>
    <tr>
      <th>126</th>
      <td>welgrd_inst5_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>5</td>
    </tr>
    <tr>
      <th>127</th>
      <td>welcst_inst6_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>6</td>
    </tr>
    <tr>
      <th>128</th>
      <td>welgrd_inst6_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>6</td>
    </tr>
    <tr>
      <th>129</th>
      <td>welcst_inst7_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>7</td>
    </tr>
    <tr>
      <th>130</th>
      <td>welgrd_inst7_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>7</td>
    </tr>
    <tr>
      <th>131</th>
      <td>welcst_inst8_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>8</td>
    </tr>
    <tr>
      <th>132</th>
      <td>welgrd_inst8_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>8</td>
    </tr>
    <tr>
      <th>133</th>
      <td>welcst_inst9_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>9</td>
    </tr>
    <tr>
      <th>134</th>
      <td>welgrd_inst9_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>9</td>
    </tr>
    <tr>
      <th>135</th>
      <td>welcst_inst10_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>10</td>
    </tr>
    <tr>
      <th>136</th>
      <td>welgrd_inst10_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>10</td>
    </tr>
    <tr>
      <th>137</th>
      <td>welcst_inst11_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>11</td>
    </tr>
    <tr>
      <th>138</th>
      <td>welgrd_inst11_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>11</td>
    </tr>
    <tr>
      <th>139</th>
      <td>welcst_inst12_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>12</td>
    </tr>
    <tr>
      <th>140</th>
      <td>welgrd_inst12_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>12</td>
    </tr>
    <tr>
      <th>141</th>
      <td>welcst_inst13_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>13</td>
    </tr>
    <tr>
      <th>142</th>
      <td>welgrd_inst13_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>13</td>
    </tr>
    <tr>
      <th>143</th>
      <td>welcst_inst14_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>14</td>
    </tr>
    <tr>
      <th>144</th>
      <td>welgrd_inst14_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>14</td>
    </tr>
    <tr>
      <th>145</th>
      <td>welcst_inst15_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>15</td>
    </tr>
    <tr>
      <th>146</th>
      <td>welgrd_inst15_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>15</td>
    </tr>
    <tr>
      <th>147</th>
      <td>welcst_inst16_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>16</td>
    </tr>
    <tr>
      <th>148</th>
      <td>welgrd_inst16_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>16</td>
    </tr>
    <tr>
      <th>149</th>
      <td>welcst_inst17_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>17</td>
    </tr>
    <tr>
      <th>150</th>
      <td>welgrd_inst17_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>17</td>
    </tr>
    <tr>
      <th>151</th>
      <td>welcst_inst18_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>18</td>
    </tr>
    <tr>
      <th>152</th>
      <td>welgrd_inst18_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>18</td>
    </tr>
    <tr>
      <th>153</th>
      <td>welcst_inst19_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>19</td>
    </tr>
    <tr>
      <th>154</th>
      <td>welgrd_inst19_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>19</td>
    </tr>
    <tr>
      <th>155</th>
      <td>welcst_inst20_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>20</td>
    </tr>
    <tr>
      <th>156</th>
      <td>welgrd_inst20_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>20</td>
    </tr>
    <tr>
      <th>157</th>
      <td>welcst_inst21_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>21</td>
    </tr>
    <tr>
      <th>158</th>
      <td>welgrd_inst21_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>21</td>
    </tr>
    <tr>
      <th>159</th>
      <td>welcst_inst22_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>22</td>
    </tr>
    <tr>
      <th>160</th>
      <td>welgrd_inst22_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>22</td>
    </tr>
    <tr>
      <th>161</th>
      <td>welcst_inst23_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>23</td>
    </tr>
    <tr>
      <th>162</th>
      <td>welgrd_inst23_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>23</td>
    </tr>
    <tr>
      <th>163</th>
      <td>welcst_inst24_constant.csv.tpl</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>24</td>
    </tr>
    <tr>
      <th>164</th>
      <td>welgrd_inst24_grid.csv.tpl</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



And the same for the template files associated with the RCH package time-varying parameters


```python
rchdf = df.loc[df.pest_file.apply(lambda x: "rch" in x and "tcn" in x),:]
df.loc[rchdf.index,"cycle"] = rchdf.pest_file.apply(lambda x: int(x.split("tcn")[0].split("_")[-1])-1)
df.loc[rchdf.index,"model_file"] = rchdf.model_file.iloc[0]
df.loc[rchdf.index,:].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pest_file</th>
      <th>model_file</th>
      <th>cycle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86</th>
      <td>rch_recharge_1tcn_inst0_constant.csv.tpl</td>
      <td>mult\rch_recharge_1tcn_inst0_constant.csv</td>
      <td>0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>rch_recharge_2tcn_inst0_constant.csv.tpl</td>
      <td>mult\rch_recharge_1tcn_inst0_constant.csv</td>
      <td>1</td>
    </tr>
    <tr>
      <th>88</th>
      <td>rch_recharge_3tcn_inst0_constant.csv.tpl</td>
      <td>mult\rch_recharge_1tcn_inst0_constant.csv</td>
      <td>2</td>
    </tr>
    <tr>
      <th>89</th>
      <td>rch_recharge_4tcn_inst0_constant.csv.tpl</td>
      <td>mult\rch_recharge_1tcn_inst0_constant.csv</td>
      <td>3</td>
    </tr>
    <tr>
      <th>90</th>
      <td>rch_recharge_5tcn_inst0_constant.csv.tpl</td>
      <td>mult\rch_recharge_1tcn_inst0_constant.csv</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



Now we need to set the cycle numbers for the parmaeters themselves - good luck doing this with recarrays!


```python
par = pst.parameter_data
par.loc[:,"cycle"] = -1
```

time-varying well parameters - the parmaeter instance ("inst") value assigned by `PstFrom` correspond to the zero-based stress period number, so we can just use that as the cycle value - nice!


```python
wpar = par.loc[par.parnme.str.contains("wel"),:]
par.loc[wpar.index,"cycle"] = wpar.inst.astype(int)
```

Same for sfr time-varying parameters:


```python
spar = par.loc[par.parnme.apply(lambda x: "sfr" in x and "cond" not in x),:]
par.loc[spar.index,"cycle"] = spar.inst.astype(int)
```

And the same for time-varying recharge parameters


```python
rpar = par.loc[par.parnme.apply(lambda x: "rch" in x and "tcn" in x),:]
par.loc[rpar.index,"cycle"] = rpar.parnme.apply(lambda x: int(x.split("tcn")[0].split("_")[-1])-1)

```

Now we need to add a special parameter that will be used to control the length of the stress period that the single-stress-period model will simulate.  As usual, we do this with a template file:


```python
with open(os.path.join(t_d,"freyberg6.tdis.tpl"),'w') as f:
    f.write("ptf ~\n")
    f.write("# new tdis written hastily at {0}\n]\n".format(datetime.now()))
    f.write("BEGIN options\n  TIME_UNITS days\nEND options\n\n")
    f.write("BEGIN dimensions\n  NPER 1\nEND dimensions\n\n")
    f.write("BEGIN perioddata\n  ~  perlen  ~  1 1.0\nEND perioddata\n\n")
```


```python
pst.add_parameters(os.path.join(t_d,"freyberg6.tdis.tpl"),pst_path=".")
```

    1 pars added from template file .\freyberg6.tdis.tpl
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parnme</th>
      <th>partrans</th>
      <th>parchglim</th>
      <th>parval1</th>
      <th>parlbnd</th>
      <th>parubnd</th>
      <th>pargp</th>
      <th>scale</th>
      <th>offset</th>
      <th>dercom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>perlen</th>
      <td>perlen</td>
      <td>log</td>
      <td>factor</td>
      <td>1.0</td>
      <td>1.100000e-10</td>
      <td>1.100000e+10</td>
      <td>pargp</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's also add a dummy parameter that is the cycle number - this will be written into the working dir at runtime and can help us get our pre and post processors going for sequential estimation


```python
tpl_file = os.path.join(t_d,"cycle.dat.tpl")
with open(tpl_file,'w') as f:
    f.write("ptf ~\n")
    f.write("cycle_num ~  cycle_num   ~\n")
pst.add_parameters(tpl_file,pst_path=".")
```

    1 pars added from template file .\cycle.dat.tpl
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parnme</th>
      <th>partrans</th>
      <th>parchglim</th>
      <th>parval1</th>
      <th>parlbnd</th>
      <th>parubnd</th>
      <th>pargp</th>
      <th>scale</th>
      <th>offset</th>
      <th>dercom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cycle_num</th>
      <td>cycle_num</td>
      <td>log</td>
      <td>factor</td>
      <td>1.0</td>
      <td>1.100000e-10</td>
      <td>1.100000e+10</td>
      <td>pargp</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pst.parameter_data.loc["perlen","partrans"] = "fixed"
pst.parameter_data.loc["perlen","cycle"] = -1
pst.parameter_data.loc["cycle_num","partrans"] = "fixed"
pst.parameter_data.loc["cycle_num","cycle"] = -1
pst.model_input_data.loc[pst.model_input_data.index[-2],"cycle"] = -1
pst.model_input_data.loc[pst.model_input_data.index[-1],"cycle"] = -1
pst.model_input_data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pest_file</th>
      <th>model_file</th>
      <th>cycle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>192</th>
      <td>freyberg6.ic_strt_layer1.txt.tpl</td>
      <td>org\freyberg6.ic_strt_layer1.txt</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>193</th>
      <td>freyberg6.ic_strt_layer2.txt.tpl</td>
      <td>org\freyberg6.ic_strt_layer2.txt</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>194</th>
      <td>freyberg6.ic_strt_layer3.txt.tpl</td>
      <td>org\freyberg6.ic_strt_layer3.txt</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>.\freyberg6.tdis.tpl</th>
      <td>.\freyberg6.tdis.tpl</td>
      <td>.\freyberg6.tdis</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>.\cycle.dat.tpl</th>
      <td>.\cycle.dat.tpl</td>
      <td>.\cycle.dat</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
</div>



Since `perlen` needs to change over cycles, we a way to tell PESTPP-DA to change it.  We could setup separate parameters and template for each cycle (e.g. `perlen_0`,`perlen_1`,`perlen_2`, etc, for cycle 0,1,2, etc), but this is cumbersome.  Instead, we can use a parameter cycle table to specific the value of the `perlen` parameter for each cycle (only fixed parameters can be treated this way...):


```python
sim = flopy.mf6.MFSimulation.load(sim_ws=org_t_d,load_only=["dis"])
org_perlen = sim.tdis.perioddata.array["perlen"]
org_perlen
```

    loading simulation...
      loading simulation name file...
      loading tdis package...
      loading model gwf6...
        loading package dis...
        skipping package ic...
        skipping package npf...
        skipping package sto...
        skipping package oc...
        skipping package wel...
        skipping package rch...
        skipping package ghb...
        skipping package sfr...
        skipping package obs...
        skipping package ims6...
    




    array([3652.5,   31. ,   29. ,   31. ,   30. ,   31. ,   30. ,   31. ,
             31. ,   30. ,   31. ,   30. ,   31. ,   31. ,   28. ,   31. ,
             30. ,   31. ,   30. ,   31. ,   31. ,   30. ,   31. ,   30. ,
             31. ])




```python
df = pd.DataFrame({"perlen":org_perlen},index=np.arange(org_perlen.shape[0]))
df.loc[:,"cycle_num"] = df.index.values
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>perlen</th>
      <th>cycle_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3652.5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>31.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30.0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>31.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>31.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>30.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>31.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>30.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>31.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>31.0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>14</th>
      <td>28.0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>15</th>
      <td>31.0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>16</th>
      <td>30.0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>17</th>
      <td>31.0</td>
      <td>17</td>
    </tr>
    <tr>
      <th>18</th>
      <td>30.0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>19</th>
      <td>31.0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>20</th>
      <td>31.0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>21</th>
      <td>30.0</td>
      <td>21</td>
    </tr>
    <tr>
      <th>22</th>
      <td>31.0</td>
      <td>22</td>
    </tr>
    <tr>
      <th>23</th>
      <td>30.0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>24</th>
      <td>31.0</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.T.to_csv(os.path.join(t_d,"par_cycle_table.csv"))
pst.pestpp_options["da_parameter_cycle_table"] = "par_cycle_table.csv"
```

Now for the observation data - yuck!  In the existing PEST interface, we include simulated GW level values in all active cells as observations, but then we also used the MF6 head obs process to make it easier for us to get the obs v sim process setup.  Here, we will ditch the MF6 head obs process and just rely on the arrays of simulated GW levels.  These important outputs will serve two roles: outputs to compare with data for assimilation (at specific locations in space and time) and also as dynamic states that will be linked to the initial head parameters - this is where things will get really exciting...


```python
obs = pst.observation_data
obs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obsnme</th>
      <th>obsval</th>
      <th>weight</th>
      <th>obgnme</th>
      <th>oname</th>
      <th>otype</th>
      <th>usecol</th>
      <th>time</th>
      <th>i</th>
      <th>j</th>
      <th>totim</th>
      <th>observed</th>
    </tr>
    <tr>
      <th>obsnme</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3652.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3652.5</td>
      <td>34.190167</td>
      <td>0.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-13-10</td>
      <td>3652.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3683.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3683.5</td>
      <td>34.178076</td>
      <td>0.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-13-10</td>
      <td>3683.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3712.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3712.5</td>
      <td>34.203032</td>
      <td>0.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-13-10</td>
      <td>3712.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3743.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3743.5</td>
      <td>34.274843</td>
      <td>0.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-13-10</td>
      <td>3743.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3773.5</th>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10_time:3773.5</td>
      <td>34.345650</td>
      <td>0.0</td>
      <td>oname:hds_otype:lst_usecol:trgw-0-13-10</td>
      <td>hds</td>
      <td>lst</td>
      <td>trgw-0-13-10</td>
      <td>3773.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4322.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4322.5</td>
      <td>0.005765</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-9-1</td>
      <td>4322.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4352.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4352.5</td>
      <td>0.005441</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-9-1</td>
      <td>4352.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4383.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:4383.5</td>
      <td>0.005251</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-9-1</td>
      <td>4383.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>part_status</th>
      <td>part_status</td>
      <td>3.000000</td>
      <td>0.0</td>
      <td>part</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>part_time</th>
      <td>part_time</td>
      <td>583999.484800</td>
      <td>0.0</td>
      <td>part</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>62227 rows × 12 columns</p>
</div>




```python
pst.model_output_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pest_file</th>
      <th>model_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>heads.csv.ins</td>
      <td>heads.csv</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sfr.csv.ins</td>
      <td>sfr.csv</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hdslay1_t1.txt.ins</td>
      <td>hdslay1_t1.txt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hdslay1_t10.txt.ins</td>
      <td>hdslay1_t10.txt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hdslay1_t11.txt.ins</td>
      <td>hdslay1_t11.txt</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>cum.csv.ins</td>
      <td>cum.csv</td>
    </tr>
    <tr>
      <th>79</th>
      <td>sfr.tdiff.csv.ins</td>
      <td>sfr.tdiff.csv</td>
    </tr>
    <tr>
      <th>80</th>
      <td>heads.tdiff.csv.ins</td>
      <td>heads.tdiff.csv</td>
    </tr>
    <tr>
      <th>81</th>
      <td>heads.vdiff.csv.ins</td>
      <td>heads.vdiff.csv</td>
    </tr>
    <tr>
      <th>82</th>
      <td>.\freyberg_mp.mpend.ins</td>
      <td>.\freyberg_mp.mpend</td>
    </tr>
  </tbody>
</table>
<p>83 rows × 2 columns</p>
</div>



Unfortunately, there is not an easy way to carry the particle-based forecasts, so let's drop those...


```python
pst.drop_observations(os.path.join(t_d,"freyberg_mp.mpend.ins"),pst_path=".")
```

    2 obs dropped from instruction file freyberg6_da_template\freyberg_mp.mpend.ins
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obsnme</th>
      <th>obsval</th>
      <th>weight</th>
      <th>obgnme</th>
      <th>oname</th>
      <th>otype</th>
      <th>usecol</th>
      <th>time</th>
      <th>i</th>
      <th>j</th>
      <th>totim</th>
      <th>observed</th>
    </tr>
    <tr>
      <th>obsnme</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>part_time</th>
      <td>part_time</td>
      <td>583999.4848</td>
      <td>0.0</td>
      <td>part</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>part_status</th>
      <td>part_status</td>
      <td>3.0000</td>
      <td>0.0</td>
      <td>part</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Same for temporal-based difference observations....


```python
pst.drop_observations(os.path.join(t_d,"sfr.tdiff.csv.ins"),pst_path=".")
```

    75 obs dropped from instruction file freyberg6_da_template\sfr.tdiff.csv.ins
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obsnme</th>
      <th>obsval</th>
      <th>weight</th>
      <th>obgnme</th>
      <th>oname</th>
      <th>otype</th>
      <th>usecol</th>
      <th>time</th>
      <th>i</th>
      <th>j</th>
      <th>totim</th>
      <th>observed</th>
    </tr>
    <tr>
      <th>obsnme</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>oname:sfrtd_otype:lst_usecol:gage-1_time:4169.5</th>
      <td>oname:sfrtd_otype:lst_usecol:gage-1_time:4169.5</td>
      <td>845.752032</td>
      <td>0.00</td>
      <td>oname:sfrtd_otype:lst_usecol:gage-1</td>
      <td>sfrtd</td>
      <td>lst</td>
      <td>gage-1</td>
      <td>4169.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:sfrtd_otype:lst_usecol:headwater_time:3987.5</th>
      <td>oname:sfrtd_otype:lst_usecol:headwater_time:3987.5</td>
      <td>377.369968</td>
      <td>0.00</td>
      <td>oname:sfrtd_otype:lst_usecol:headwater</td>
      <td>sfrtd</td>
      <td>lst</td>
      <td>headwater</td>
      <td>3987.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:sfrtd_otype:lst_usecol:gage-1_time:3773.5</th>
      <td>oname:sfrtd_otype:lst_usecol:gage-1_time:3773.5</td>
      <td>842.029821</td>
      <td>0.01</td>
      <td>oname:sfrtd_otype:lst_usecol:gage-1</td>
      <td>sfrtd</td>
      <td>lst</td>
      <td>gage-1</td>
      <td>3773.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>oname:sfrtd_otype:lst_usecol:tailwater_time:3896.5</th>
      <td>oname:sfrtd_otype:lst_usecol:tailwater_time:3896.5</td>
      <td>298.651545</td>
      <td>0.00</td>
      <td>oname:sfrtd_otype:lst_usecol:tailwater</td>
      <td>sfrtd</td>
      <td>lst</td>
      <td>tailwater</td>
      <td>3896.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:sfrtd_otype:lst_usecol:tailwater_time:4199.5</th>
      <td>oname:sfrtd_otype:lst_usecol:tailwater_time:4199.5</td>
      <td>60.565373</td>
      <td>0.00</td>
      <td>oname:sfrtd_otype:lst_usecol:tailwater</td>
      <td>sfrtd</td>
      <td>lst</td>
      <td>tailwater</td>
      <td>4199.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>oname:sfrtd_otype:lst_usecol:tailwater_time:4230.5</th>
      <td>oname:sfrtd_otype:lst_usecol:tailwater_time:4230.5</td>
      <td>223.737212</td>
      <td>0.00</td>
      <td>oname:sfrtd_otype:lst_usecol:tailwater</td>
      <td>sfrtd</td>
      <td>lst</td>
      <td>tailwater</td>
      <td>4230.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:sfrtd_otype:lst_usecol:gage-1_time:3957.5</th>
      <td>oname:sfrtd_otype:lst_usecol:gage-1_time:3957.5</td>
      <td>-370.470970</td>
      <td>0.01</td>
      <td>oname:sfrtd_otype:lst_usecol:gage-1</td>
      <td>sfrtd</td>
      <td>lst</td>
      <td>gage-1</td>
      <td>3957.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>oname:sfrtd_otype:lst_usecol:headwater_time:4049.5</th>
      <td>oname:sfrtd_otype:lst_usecol:headwater_time:4049.5</td>
      <td>-70.344875</td>
      <td>0.00</td>
      <td>oname:sfrtd_otype:lst_usecol:headwater</td>
      <td>sfrtd</td>
      <td>lst</td>
      <td>headwater</td>
      <td>4049.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:sfrtd_otype:lst_usecol:headwater_time:3957.5</th>
      <td>oname:sfrtd_otype:lst_usecol:headwater_time:3957.5</td>
      <td>504.114353</td>
      <td>0.00</td>
      <td>oname:sfrtd_otype:lst_usecol:headwater</td>
      <td>sfrtd</td>
      <td>lst</td>
      <td>headwater</td>
      <td>3957.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:sfrtd_otype:lst_usecol:gage-1_time:3804.5</th>
      <td>oname:sfrtd_otype:lst_usecol:gage-1_time:3804.5</td>
      <td>1090.165816</td>
      <td>0.01</td>
      <td>oname:sfrtd_otype:lst_usecol:gage-1</td>
      <td>sfrtd</td>
      <td>lst</td>
      <td>gage-1</td>
      <td>3804.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 12 columns</p>
</div>




```python
pst.drop_observations(os.path.join(t_d,"heads.tdiff.csv.ins"),pst_path=".")
pst.drop_observations(os.path.join(t_d,"heads.vdiff.csv.ins"),pst_path=".")
```

    650 obs dropped from instruction file freyberg6_da_template\heads.tdiff.csv.ins
    325 obs dropped from instruction file freyberg6_da_template\heads.vdiff.csv.ins
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obsnme</th>
      <th>obsval</th>
      <th>weight</th>
      <th>obgnme</th>
      <th>oname</th>
      <th>otype</th>
      <th>usecol</th>
      <th>time</th>
      <th>i</th>
      <th>j</th>
      <th>totim</th>
      <th>observed</th>
    </tr>
    <tr>
      <th>obsnme</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-29-15_time:4199.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-29-15_time:4199.5</td>
      <td>-0.034342</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-29-15</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-29-15</td>
      <td>4199.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-29-15_time:4261.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-29-15_time:4261.5</td>
      <td>-0.012459</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-29-15</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-29-15</td>
      <td>4261.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-34-10_time:3712.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-34-10_time:3712.5</td>
      <td>0.006580</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-34-10</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-34-10</td>
      <td>3712.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-34-10_time:3743.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-34-10_time:3743.5</td>
      <td>0.007652</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-34-10</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-34-10</td>
      <td>3743.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:3773.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1_time:3773.5</td>
      <td>0.007117</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-9-1</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-9-1</td>
      <td>3773.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-34-10_time:3926.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-34-10_time:3926.5</td>
      <td>0.004009</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-34-10</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-34-10</td>
      <td>3926.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-22-15_time:4352.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-22-15_time:4352.5</td>
      <td>-0.024070</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-22-15</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-22-15</td>
      <td>4352.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-3-8_time:3683.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-3-8_time:3683.5</td>
      <td>0.028698</td>
      <td>100.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-3-8</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-3-8</td>
      <td>3683.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-22-15_time:3834.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-22-15_time:3834.5</td>
      <td>-0.039959</td>
      <td>0.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-22-15</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-22-15</td>
      <td>3834.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>oname:hdsvd_otype:lst_usecol:trgw-0-3-8_time:3926.5</th>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-3-8_time:3926.5</td>
      <td>-0.029573</td>
      <td>100.0</td>
      <td>oname:hdsvd_otype:lst_usecol:trgw-0-3-8</td>
      <td>hdsvd</td>
      <td>lst</td>
      <td>trgw-0-3-8</td>
      <td>3926.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>325 rows × 12 columns</p>
</div>



Here is where we will drop the MF6 head obs type observations - remember, we will instead rely on the arrays of simulated GW levels


```python
hdf = pst.drop_observations(os.path.join(t_d,"heads.csv.ins"),pst_path=".")

#sdf = pst.drop_observations(os.path.join(t_d,"sfr.csv.ins"),pst_path=".")
```

    650 obs dropped from instruction file freyberg6_da_template\heads.csv.ins
    


```python
#[i for i in pst.model_output_data.model_file if i.startswith('hdslay')]
```


```python
pst.model_output_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pest_file</th>
      <th>model_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>sfr.csv.ins</td>
      <td>sfr.csv</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hdslay1_t1.txt.ins</td>
      <td>hdslay1_t1.txt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>hdslay1_t10.txt.ins</td>
      <td>hdslay1_t10.txt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hdslay1_t11.txt.ins</td>
      <td>hdslay1_t11.txt</td>
    </tr>
    <tr>
      <th>5</th>
      <td>hdslay1_t12.txt.ins</td>
      <td>hdslay1_t12.txt</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>74</th>
      <td>hdslay3_t7.txt.ins</td>
      <td>hdslay3_t7.txt</td>
    </tr>
    <tr>
      <th>75</th>
      <td>hdslay3_t8.txt.ins</td>
      <td>hdslay3_t8.txt</td>
    </tr>
    <tr>
      <th>76</th>
      <td>hdslay3_t9.txt.ins</td>
      <td>hdslay3_t9.txt</td>
    </tr>
    <tr>
      <th>77</th>
      <td>inc.csv.ins</td>
      <td>inc.csv</td>
    </tr>
    <tr>
      <th>78</th>
      <td>cum.csv.ins</td>
      <td>cum.csv</td>
    </tr>
  </tbody>
</table>
<p>78 rows × 2 columns</p>
</div>



Now for some really nasty hackery:  we are going to modify the remaining stress-period-based instruction files to only include one row of instructions (since we only have one stress period now):


```python
sfrdf = None
for ins_file in pst.model_output_data.pest_file:
    if ins_file.startswith("hdslay") and ins_file.endswith("_t1.txt.ins"):
        print('not dropping:', ins_file)
        continue
    elif ins_file.startswith("hdslay"):
        df = pst.drop_observations(os.path.join(t_d,ins_file),pst_path=".")
        print('dropping:',ins_file)
    else:
        lines = open(os.path.join(t_d,ins_file),'r').readlines()
        df = pst.drop_observations(os.path.join(t_d,ins_file),pst_path=".")
        if ins_file == "sfr.csv.ins":
            sfrdf = df
        with open(os.path.join(t_d,ins_file),'w') as f:
            for line in lines[:3]:
                f.write(line.replace("_totim:3652.5","").replace("_time:3652.5",""))
        pst.add_observations(os.path.join(t_d,ins_file),pst_path=".")
assert sfrdf is not None
```

    75 obs dropped from instruction file freyberg6_da_template\sfr.csv.ins
    3 obs added from instruction file freyberg6_da_template\.\sfr.csv.ins
    not dropping: hdslay1_t1.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t10.txt.ins
    dropping: hdslay1_t10.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t11.txt.ins
    dropping: hdslay1_t11.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t12.txt.ins
    dropping: hdslay1_t12.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t13.txt.ins
    dropping: hdslay1_t13.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t14.txt.ins
    dropping: hdslay1_t14.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t15.txt.ins
    dropping: hdslay1_t15.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t16.txt.ins
    dropping: hdslay1_t16.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t17.txt.ins
    dropping: hdslay1_t17.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t18.txt.ins
    dropping: hdslay1_t18.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t19.txt.ins
    dropping: hdslay1_t19.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t2.txt.ins
    dropping: hdslay1_t2.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t20.txt.ins
    dropping: hdslay1_t20.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t21.txt.ins
    dropping: hdslay1_t21.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t22.txt.ins
    dropping: hdslay1_t22.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t23.txt.ins
    dropping: hdslay1_t23.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t24.txt.ins
    dropping: hdslay1_t24.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t25.txt.ins
    dropping: hdslay1_t25.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t3.txt.ins
    dropping: hdslay1_t3.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t4.txt.ins
    dropping: hdslay1_t4.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t5.txt.ins
    dropping: hdslay1_t5.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t6.txt.ins
    dropping: hdslay1_t6.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t7.txt.ins
    dropping: hdslay1_t7.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t8.txt.ins
    dropping: hdslay1_t8.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay1_t9.txt.ins
    dropping: hdslay1_t9.txt.ins
    not dropping: hdslay2_t1.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t10.txt.ins
    dropping: hdslay2_t10.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t11.txt.ins
    dropping: hdslay2_t11.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t12.txt.ins
    dropping: hdslay2_t12.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t13.txt.ins
    dropping: hdslay2_t13.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t14.txt.ins
    dropping: hdslay2_t14.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t15.txt.ins
    dropping: hdslay2_t15.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t16.txt.ins
    dropping: hdslay2_t16.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t17.txt.ins
    dropping: hdslay2_t17.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t18.txt.ins
    dropping: hdslay2_t18.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t19.txt.ins
    dropping: hdslay2_t19.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t2.txt.ins
    dropping: hdslay2_t2.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t20.txt.ins
    dropping: hdslay2_t20.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t21.txt.ins
    dropping: hdslay2_t21.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t22.txt.ins
    dropping: hdslay2_t22.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t23.txt.ins
    dropping: hdslay2_t23.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t24.txt.ins
    dropping: hdslay2_t24.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t25.txt.ins
    dropping: hdslay2_t25.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t3.txt.ins
    dropping: hdslay2_t3.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t4.txt.ins
    dropping: hdslay2_t4.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t5.txt.ins
    dropping: hdslay2_t5.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t6.txt.ins
    dropping: hdslay2_t6.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t7.txt.ins
    dropping: hdslay2_t7.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t8.txt.ins
    dropping: hdslay2_t8.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay2_t9.txt.ins
    dropping: hdslay2_t9.txt.ins
    not dropping: hdslay3_t1.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t10.txt.ins
    dropping: hdslay3_t10.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t11.txt.ins
    dropping: hdslay3_t11.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t12.txt.ins
    dropping: hdslay3_t12.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t13.txt.ins
    dropping: hdslay3_t13.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t14.txt.ins
    dropping: hdslay3_t14.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t15.txt.ins
    dropping: hdslay3_t15.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t16.txt.ins
    dropping: hdslay3_t16.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t17.txt.ins
    dropping: hdslay3_t17.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t18.txt.ins
    dropping: hdslay3_t18.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t19.txt.ins
    dropping: hdslay3_t19.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t2.txt.ins
    dropping: hdslay3_t2.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t20.txt.ins
    dropping: hdslay3_t20.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t21.txt.ins
    dropping: hdslay3_t21.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t22.txt.ins
    dropping: hdslay3_t22.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t23.txt.ins
    dropping: hdslay3_t23.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t24.txt.ins
    dropping: hdslay3_t24.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t25.txt.ins
    dropping: hdslay3_t25.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t3.txt.ins
    dropping: hdslay3_t3.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t4.txt.ins
    dropping: hdslay3_t4.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t5.txt.ins
    dropping: hdslay3_t5.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t6.txt.ins
    dropping: hdslay3_t6.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t7.txt.ins
    dropping: hdslay3_t7.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t8.txt.ins
    dropping: hdslay3_t8.txt.ins
    800 obs dropped from instruction file freyberg6_da_template\hdslay3_t9.txt.ins
    dropping: hdslay3_t9.txt.ins
    225 obs dropped from instruction file freyberg6_da_template\inc.csv.ins
    9 obs added from instruction file freyberg6_da_template\.\inc.csv.ins
    225 obs dropped from instruction file freyberg6_da_template\cum.csv.ins
    9 obs added from instruction file freyberg6_da_template\.\cum.csv.ins
    


```python
[i for i in pst.model_output_data.model_file if i.startswith('hdslay')]
```




    ['hdslay1_t1.txt', 'hdslay2_t1.txt', 'hdslay3_t1.txt']




```python
pst.model_output_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pest_file</th>
      <th>model_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>hdslay1_t1.txt.ins</td>
      <td>hdslay1_t1.txt</td>
    </tr>
    <tr>
      <th>27</th>
      <td>hdslay2_t1.txt.ins</td>
      <td>hdslay2_t1.txt</td>
    </tr>
    <tr>
      <th>52</th>
      <td>hdslay3_t1.txt.ins</td>
      <td>hdslay3_t1.txt</td>
    </tr>
    <tr>
      <th>.\sfr.csv.ins</th>
      <td>.\sfr.csv.ins</td>
      <td>.\sfr.csv</td>
    </tr>
    <tr>
      <th>.\inc.csv.ins</th>
      <td>.\inc.csv.ins</td>
      <td>.\inc.csv</td>
    </tr>
    <tr>
      <th>.\cum.csv.ins</th>
      <td>.\cum.csv.ins</td>
      <td>.\cum.csv</td>
    </tr>
  </tbody>
</table>
</div>



Time to work out a mapping from the MF6 head obs data (that have the actual head observations and weight we want) to the array based GW level observations.  We will again use a special set of PESTPP-DA specific options to help us here.  Since the observed value of GW level and the weights change through time (e.g. across cycles) but we are recording the array-based GW level observations every cycle, we need a way to tell PESTPP-DA to use specific `obsval`s and `weight`s for a given cycle.  `da_observation_cycle_table` and `da_weight_cycle_table` to the rescue!


```python
hdf.loc[:,"k"] = hdf.usecol.apply(lambda x: int(x.split("-")[1]))
hdf.loc[:,"i"] = hdf.usecol.apply(lambda x: int(x.split("-")[2]))
hdf.loc[:,"j"] = hdf.usecol.apply(lambda x: int(x.split("-")[3]))
hdf.loc[:,"time"] = hdf.time.astype(float)
```


```python
sites = hdf.usecol.unique()
sites.sort()
sites
```




    array(['trgw-0-13-10', 'trgw-0-15-16', 'trgw-0-2-15', 'trgw-0-2-9',
           'trgw-0-21-10', 'trgw-0-22-15', 'trgw-0-24-4', 'trgw-0-26-6',
           'trgw-0-29-15', 'trgw-0-3-8', 'trgw-0-33-7', 'trgw-0-34-10',
           'trgw-0-9-1', 'trgw-2-13-10', 'trgw-2-15-16', 'trgw-2-2-15',
           'trgw-2-2-9', 'trgw-2-21-10', 'trgw-2-22-15', 'trgw-2-24-4',
           'trgw-2-26-6', 'trgw-2-29-15', 'trgw-2-3-8', 'trgw-2-33-7',
           'trgw-2-34-10', 'trgw-2-9-1'], dtype=object)



In this code bit, we will process each MF6 head obs record (which includes `obsval` and `weight` for each stress period at each L-R-C location) and align that with corresponding (L-R-C) array-based GW level observation.  Then just collate those records into obs and weight cycle table. Note: we only want to include sites that have at least one non-zero weighted observation.  easy as!


```python
pst.try_parse_name_metadata()
obs = pst.observation_data
hdsobs = obs.loc[obs.obsnme.str.contains("hdslay"),:].copy()
hdsobs.loc[:,"i"] = hdsobs.i.astype(int)
hdsobs.loc[:,"j"] = hdsobs.j.astype(int)
hdsobs.loc[:,"k"] = hdsobs.oname.apply(lambda x: int(x[-1])-1)
odata = {}
wdata = {}
alldata = {}
for site in sites:
    sdf = hdf.loc[hdf.usecol==site,:].copy()
    #print(sdf.weight)
    
    sdf.sort_values(by="time",inplace=True)
    k,i,j = sdf.k.iloc[0],sdf.i.iloc[0],sdf.j.iloc[0]
    hds = hdsobs.loc[hdsobs.apply(lambda x: x.i==i and x.j==j and x.k==k,axis=1),:].copy()
    #assert hds.shape[0] == 1,site
    obname = hds.obsnme.iloc[0]
    print(obname)
    alldata[obname] = sdf.obsval.values
    if sdf.weight.sum() == 0:
        continue
    odata[obname] = sdf.obsval.values
    wdata[obname] = sdf.weight.values
    #print(site)
        
```

    oname:hdslay1_t1_otype:arr_i:13_j:10
    oname:hdslay1_t1_otype:arr_i:15_j:16
    oname:hdslay1_t1_otype:arr_i:2_j:15
    oname:hdslay1_t1_otype:arr_i:2_j:9
    oname:hdslay1_t1_otype:arr_i:21_j:10
    oname:hdslay1_t1_otype:arr_i:22_j:15
    oname:hdslay1_t1_otype:arr_i:24_j:4
    oname:hdslay1_t1_otype:arr_i:26_j:6
    oname:hdslay1_t1_otype:arr_i:29_j:15
    oname:hdslay1_t1_otype:arr_i:3_j:8
    oname:hdslay1_t1_otype:arr_i:33_j:7
    oname:hdslay1_t1_otype:arr_i:34_j:10
    oname:hdslay1_t1_otype:arr_i:9_j:1
    oname:hdslay3_t1_otype:arr_i:13_j:10
    oname:hdslay3_t1_otype:arr_i:15_j:16
    oname:hdslay3_t1_otype:arr_i:2_j:15
    oname:hdslay3_t1_otype:arr_i:2_j:9
    oname:hdslay3_t1_otype:arr_i:21_j:10
    oname:hdslay3_t1_otype:arr_i:22_j:15
    oname:hdslay3_t1_otype:arr_i:24_j:4
    oname:hdslay3_t1_otype:arr_i:26_j:6
    oname:hdslay3_t1_otype:arr_i:29_j:15
    oname:hdslay3_t1_otype:arr_i:3_j:8
    oname:hdslay3_t1_otype:arr_i:33_j:7
    oname:hdslay3_t1_otype:arr_i:34_j:10
    oname:hdslay3_t1_otype:arr_i:9_j:1
    

Same for the SFR "gage-1" observations


```python
sfrobs = obs.loc[obs.obsnme.str.contains("oname:sfr"),:].copy()
sites = sfrdf.usecol.unique()
sites.sort()
sites
```




    array(['gage-1', 'headwater', 'tailwater'], dtype=object)




```python
for site in sites:
    sdf = sfrdf.loc[sfrdf.usecol==site,:].copy()
    sdf.loc[:,"time"] = sdf.time.astype(float)
    
    sdf.sort_values(by="time",inplace=True)
    sfr = sfrobs.loc[sfrobs.usecol==site,:].copy()
    assert sfr.shape[0] == 1,sfr
    alldata[sfr.obsnme.iloc[0]] = sdf.obsval.values
    if sdf.weight.sum() == 0:
        continue
    odata[sfr.obsnme.iloc[0]] = sdf.obsval.values
    wdata[sfr.obsnme.iloc[0]] = sdf.weight.values
        
```


```python
odata
```




    {'oname:hdslay1_t1_otype:arr_i:26_j:6': array([34.64401284, 34.55417855, 34.63951738, 34.79799406, 34.92316856,
            35.0804532 , 35.12298786, 35.11143536, 34.98346193, 34.80007573,
            34.51338982, 34.40557368, 34.35630664, 34.35998071, 34.43775967,
            34.6127413 , 34.78599495, 34.92919168, 34.97556532, 34.98548115,
            34.91409599, 34.72213818, 34.55599691, 34.38654068, 34.27659649]),
     'oname:hdslay1_t1_otype:arr_i:3_j:8': array([34.57174738, 34.58024082, 34.55952273, 34.61496256, 34.65134148,
            34.76128283, 34.78827913, 34.82060015, 34.78993691, 34.69760476,
            34.69114137, 34.62222203, 34.57216813, 34.50866346, 34.50346713,
            34.56958992, 34.60113449, 34.68088019, 34.72984612, 34.75531127,
            34.71573532, 34.67921814, 34.60283652, 34.60053219, 34.51060353]),
     'oname:hdslay3_t1_otype:arr_i:26_j:6': array([34.56549551, 34.56663841, 34.63688816, 34.79590895, 34.92812757,
            35.03681128, 35.10354503, 35.12136373, 35.01306747, 34.75767713,
            34.49140978, 34.40133231, 34.33919818, 34.33882667, 34.47976782,
            34.60365861, 34.73421213, 34.8951266 , 35.02124038, 34.99486661,
            34.93362477, 34.73630864, 34.55899385, 34.36437442, 34.340205  ]),
     'oname:hdslay3_t1_otype:arr_i:3_j:8': array([34.57998451, 34.5515424 , 34.53473414, 34.67630406, 34.66700792,
            34.76317135, 34.79316777, 34.81747467, 34.76665706, 34.72717728,
            34.6828193 , 34.63977708, 34.55981371, 34.54642867, 34.48552957,
            34.55163166, 34.63383379, 34.64655334, 34.72426798, 34.77080219,
            34.70107751, 34.68861779, 34.64658789, 34.58091824, 34.54353048]),
     'oname:sfr_otype:lst_usecol:gage-1': array([1998.42556366, 1914.07443813, 2047.80383267, 2538.18528081,
            2840.45538436, 3088.59137928, 3244.63127124, 2933.98309136,
            2592.89159622, 2006.87241144, 1627.95459359, 1492.09814392,
            1417.8828934 , 1608.66642246, 1867.78673662, 2206.52911895,
            2550.21262216, 2844.1775956 , 2971.34060367, 2753.93542131,
            2454.78493742, 2051.61964221, 1708.83548765, 1512.80823647,
            1419.77513167])}



Form the obs cycle table as a dataframe:


```python
df = pd.DataFrame(odata)
df.index.name = "cycle"
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>oname:hdslay1_t1_otype:arr_i:26_j:6</th>
      <th>oname:hdslay1_t1_otype:arr_i:3_j:8</th>
      <th>oname:hdslay3_t1_otype:arr_i:26_j:6</th>
      <th>oname:hdslay3_t1_otype:arr_i:3_j:8</th>
      <th>oname:sfr_otype:lst_usecol:gage-1</th>
    </tr>
    <tr>
      <th>cycle</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34.644013</td>
      <td>34.571747</td>
      <td>34.565496</td>
      <td>34.579985</td>
      <td>1998.425564</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34.554179</td>
      <td>34.580241</td>
      <td>34.566638</td>
      <td>34.551542</td>
      <td>1914.074438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34.639517</td>
      <td>34.559523</td>
      <td>34.636888</td>
      <td>34.534734</td>
      <td>2047.803833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34.797994</td>
      <td>34.614963</td>
      <td>34.795909</td>
      <td>34.676304</td>
      <td>2538.185281</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34.923169</td>
      <td>34.651341</td>
      <td>34.928128</td>
      <td>34.667008</td>
      <td>2840.455384</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.080453</td>
      <td>34.761283</td>
      <td>35.036811</td>
      <td>34.763171</td>
      <td>3088.591379</td>
    </tr>
    <tr>
      <th>6</th>
      <td>35.122988</td>
      <td>34.788279</td>
      <td>35.103545</td>
      <td>34.793168</td>
      <td>3244.631271</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35.111435</td>
      <td>34.820600</td>
      <td>35.121364</td>
      <td>34.817475</td>
      <td>2933.983091</td>
    </tr>
    <tr>
      <th>8</th>
      <td>34.983462</td>
      <td>34.789937</td>
      <td>35.013067</td>
      <td>34.766657</td>
      <td>2592.891596</td>
    </tr>
    <tr>
      <th>9</th>
      <td>34.800076</td>
      <td>34.697605</td>
      <td>34.757677</td>
      <td>34.727177</td>
      <td>2006.872411</td>
    </tr>
    <tr>
      <th>10</th>
      <td>34.513390</td>
      <td>34.691141</td>
      <td>34.491410</td>
      <td>34.682819</td>
      <td>1627.954594</td>
    </tr>
    <tr>
      <th>11</th>
      <td>34.405574</td>
      <td>34.622222</td>
      <td>34.401332</td>
      <td>34.639777</td>
      <td>1492.098144</td>
    </tr>
    <tr>
      <th>12</th>
      <td>34.356307</td>
      <td>34.572168</td>
      <td>34.339198</td>
      <td>34.559814</td>
      <td>1417.882893</td>
    </tr>
    <tr>
      <th>13</th>
      <td>34.359981</td>
      <td>34.508663</td>
      <td>34.338827</td>
      <td>34.546429</td>
      <td>1608.666422</td>
    </tr>
    <tr>
      <th>14</th>
      <td>34.437760</td>
      <td>34.503467</td>
      <td>34.479768</td>
      <td>34.485530</td>
      <td>1867.786737</td>
    </tr>
    <tr>
      <th>15</th>
      <td>34.612741</td>
      <td>34.569590</td>
      <td>34.603659</td>
      <td>34.551632</td>
      <td>2206.529119</td>
    </tr>
    <tr>
      <th>16</th>
      <td>34.785995</td>
      <td>34.601134</td>
      <td>34.734212</td>
      <td>34.633834</td>
      <td>2550.212622</td>
    </tr>
    <tr>
      <th>17</th>
      <td>34.929192</td>
      <td>34.680880</td>
      <td>34.895127</td>
      <td>34.646553</td>
      <td>2844.177596</td>
    </tr>
    <tr>
      <th>18</th>
      <td>34.975565</td>
      <td>34.729846</td>
      <td>35.021240</td>
      <td>34.724268</td>
      <td>2971.340604</td>
    </tr>
    <tr>
      <th>19</th>
      <td>34.985481</td>
      <td>34.755311</td>
      <td>34.994867</td>
      <td>34.770802</td>
      <td>2753.935421</td>
    </tr>
    <tr>
      <th>20</th>
      <td>34.914096</td>
      <td>34.715735</td>
      <td>34.933625</td>
      <td>34.701078</td>
      <td>2454.784937</td>
    </tr>
    <tr>
      <th>21</th>
      <td>34.722138</td>
      <td>34.679218</td>
      <td>34.736309</td>
      <td>34.688618</td>
      <td>2051.619642</td>
    </tr>
    <tr>
      <th>22</th>
      <td>34.555997</td>
      <td>34.602837</td>
      <td>34.558994</td>
      <td>34.646588</td>
      <td>1708.835488</td>
    </tr>
    <tr>
      <th>23</th>
      <td>34.386541</td>
      <td>34.600532</td>
      <td>34.364374</td>
      <td>34.580918</td>
      <td>1512.808236</td>
    </tr>
    <tr>
      <th>24</th>
      <td>34.276596</td>
      <td>34.510604</td>
      <td>34.340205</td>
      <td>34.543530</td>
      <td>1419.775132</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.T.to_csv(os.path.join(t_d,"obs_cycle_table.csv"))
pst.pestpp_options["da_observation_cycle_table"] = "obs_cycle_table.csv"
```

Prep for the weight cycle table also.  As a safety check, PESTPP-DA requires any observation quantity that ever has a non-zero weight for any cycle to have a non-zero weight in `* observation data` (this weight value is not used, its more of just a flag).


```python
obs = pst.observation_data
obs.loc[:,"weight"] = 0
obs.loc[:,"cycle"] = -1
df = pd.DataFrame(wdata)
df.index.name = "cycle"
wsum = df.sum()
wsum = wsum.loc[wsum>0]
print(wsum)
obs.loc[wsum.index,"weight"] = 1.0

df.T.to_csv(os.path.join(t_d,"weight_cycle_table.csv"))
pst.pestpp_options["da_weight_cycle_table"] = "weight_cycle_table.csv"
df
```

    oname:hdslay1_t1_otype:arr_i:26_j:6    120.000000
    oname:hdslay1_t1_otype:arr_i:3_j:8     120.000000
    oname:hdslay3_t1_otype:arr_i:26_j:6    120.000000
    oname:hdslay3_t1_otype:arr_i:3_j:8     120.000000
    oname:sfr_otype:lst_usecol:gage-1        0.056033
    dtype: float64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>oname:hdslay1_t1_otype:arr_i:26_j:6</th>
      <th>oname:hdslay1_t1_otype:arr_i:3_j:8</th>
      <th>oname:hdslay3_t1_otype:arr_i:26_j:6</th>
      <th>oname:hdslay3_t1_otype:arr_i:3_j:8</th>
      <th>oname:sfr_otype:lst_usecol:gage-1</th>
    </tr>
    <tr>
      <th>cycle</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.005224</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.004883</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.003940</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.003521</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.003238</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.003082</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.003408</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.003857</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.004983</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.006143</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.006702</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.007053</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



Nothing to see here...let's save `alldata` to help us plot the results of PESTPP-DA later WRT forecasts and un-assimilated observations


```python
df = pd.DataFrame(alldata)
df.index.name = "cycle"
df.to_csv(os.path.join(t_d,"alldata.csv"))
```

### The state mapping between pars and obs

Ok, for our next trick...

We need to tell PESTPP-DA that we want to use dynamic states.  This is tricky concept for us "batch" people, but conceptually, these states allow PESTPP-DA to coherently advance the model in time.  Just like MF6 would take the final simulated GW levels at the end of stress period and set them as the starting heads for the next stress, so too must PESTPP-DA. Otherwise, there would be no temporal coherence in the simulated results.  What is exciting about this is that PESTPP-DA also has the opportunity to "estimate" the start heads for each cycle, along with the other parameters.  Algorithmically, PESTPP-DA sees these "states" just as any other parameter to estimate for a given cycle.  Conceptually, treating the initial states for each cycle as uncertain and therefore adjustable, is one way to explicitly acknowledge that the model is "imperfect" and therefore the initial conditions for each cycle are "imperfect" e.g. uncertain!  How cool!

The way we tell PESTPP-DA about the dynamic state linkage between observations and parameters is by either giving the parameters and observations identical names, or by adding a column to the `* observation data` dataframe that names the parameter that the observation links to.  We will do the latter here - this column must be named "state_par_link":


```python
obs = pst.observation_data
obs.loc[:,"state_par_link"] = ""
hdsobs = obs.loc[obs.obsnme.str.contains("hdslay"),:].copy()
hdsobs.loc[:,"i"] = hdsobs.i.astype(int)
hdsobs.loc[:,"j"] = hdsobs.j.astype(int)
hdsobs.loc[:,"k"] = hdsobs.oname.apply(lambda x: int(x[-1])-1)
hdsobs.loc[:,"kij"] = hdsobs.apply(lambda x: (x.k,x.i,x.j),axis=1)
```


```python
par = pst.parameter_data
strtpar = par.loc[par.parnme.str.contains("strt"),:].copy()
strtpar.loc[:,"i"] = strtpar.i.astype(int)
strtpar.loc[:,"j"] = strtpar.j.astype(int)
strtpar.loc[:,"k"] = strtpar.pname.apply(lambda x: int(x[-1])-1)
strtpar.loc[:,"kij"] = strtpar.apply(lambda x: (x.k,x.i,x.j),axis=1)
spl = {kij:name for kij,name in zip(strtpar.kij,strtpar.parnme)}
```


```python
obs.loc[hdsobs.obsnme,"state_par_link"] = hdsobs.kij.apply(lambda x: spl.get(x,""))
```


```python
obs.loc[hdsobs.obsnme,:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>obsnme</th>
      <th>obsval</th>
      <th>weight</th>
      <th>obgnme</th>
      <th>oname</th>
      <th>otype</th>
      <th>usecol</th>
      <th>time</th>
      <th>i</th>
      <th>j</th>
      <th>totim</th>
      <th>observed</th>
      <th>cycle</th>
      <th>state_par_link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>oname:hdslay1_t1_otype:arr_i:0_j:0</th>
      <td>oname:hdslay1_t1_otype:arr_i:0_j:0</td>
      <td>3.500016e+01</td>
      <td>0</td>
      <td>hdslay1_t1</td>
      <td>hdslay1</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>pname:icstrtlayer1_inst:0_ptype:gr_pstyle:d_i:0_j:0_x:125.00_y:9875.00_zone:1</td>
    </tr>
    <tr>
      <th>oname:hdslay1_t1_otype:arr_i:0_j:1</th>
      <td>oname:hdslay1_t1_otype:arr_i:0_j:1</td>
      <td>3.499182e+01</td>
      <td>0</td>
      <td>hdslay1_t1</td>
      <td>hdslay1</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>pname:icstrtlayer1_inst:0_ptype:gr_pstyle:d_i:0_j:1_x:375.00_y:9875.00_zone:1</td>
    </tr>
    <tr>
      <th>oname:hdslay1_t1_otype:arr_i:0_j:10</th>
      <td>oname:hdslay1_t1_otype:arr_i:0_j:10</td>
      <td>3.466480e+01</td>
      <td>0</td>
      <td>hdslay1_t1</td>
      <td>hdslay1</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>pname:icstrtlayer1_inst:0_ptype:gr_pstyle:d_i:0_j:10_x:2625.00_y:9875.00_zone:1</td>
    </tr>
    <tr>
      <th>oname:hdslay1_t1_otype:arr_i:0_j:11</th>
      <td>oname:hdslay1_t1_otype:arr_i:0_j:11</td>
      <td>3.461760e+01</td>
      <td>0</td>
      <td>hdslay1_t1</td>
      <td>hdslay1</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>pname:icstrtlayer1_inst:0_ptype:gr_pstyle:d_i:0_j:11_x:2875.00_y:9875.00_zone:1</td>
    </tr>
    <tr>
      <th>oname:hdslay1_t1_otype:arr_i:0_j:12</th>
      <td>oname:hdslay1_t1_otype:arr_i:0_j:12</td>
      <td>3.456958e+01</td>
      <td>0</td>
      <td>hdslay1_t1</td>
      <td>hdslay1</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>pname:icstrtlayer1_inst:0_ptype:gr_pstyle:d_i:0_j:12_x:3125.00_y:9875.00_zone:1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>oname:hdslay3_t1_otype:arr_i:9_j:5</th>
      <td>oname:hdslay3_t1_otype:arr_i:9_j:5</td>
      <td>1.000000e+30</td>
      <td>0</td>
      <td>hdslay3_t1</td>
      <td>hdslay3</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td></td>
    </tr>
    <tr>
      <th>oname:hdslay3_t1_otype:arr_i:9_j:6</th>
      <td>oname:hdslay3_t1_otype:arr_i:9_j:6</td>
      <td>1.000000e+30</td>
      <td>0</td>
      <td>hdslay3_t1</td>
      <td>hdslay3</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td></td>
    </tr>
    <tr>
      <th>oname:hdslay3_t1_otype:arr_i:9_j:7</th>
      <td>oname:hdslay3_t1_otype:arr_i:9_j:7</td>
      <td>1.000000e+30</td>
      <td>0</td>
      <td>hdslay3_t1</td>
      <td>hdslay3</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
      <td>7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td></td>
    </tr>
    <tr>
      <th>oname:hdslay3_t1_otype:arr_i:9_j:8</th>
      <td>oname:hdslay3_t1_otype:arr_i:9_j:8</td>
      <td>3.450352e+01</td>
      <td>0</td>
      <td>hdslay3_t1</td>
      <td>hdslay3</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>pname:icstrtlayer3_inst:0_ptype:gr_pstyle:d_i:9_j:8_x:2125.00_y:7625.00_zone:1</td>
    </tr>
    <tr>
      <th>oname:hdslay3_t1_otype:arr_i:9_j:9</th>
      <td>oname:hdslay3_t1_otype:arr_i:9_j:9</td>
      <td>3.447937e+01</td>
      <td>0</td>
      <td>hdslay3_t1</td>
      <td>hdslay3</td>
      <td>arr</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9</td>
      <td>9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1</td>
      <td>pname:icstrtlayer3_inst:0_ptype:gr_pstyle:d_i:9_j:9_x:2375.00_y:7625.00_zone:1</td>
    </tr>
  </tbody>
</table>
<p>2400 rows × 14 columns</p>
</div>



One last thing: we need to modify the multiplier-parameter process since we now have a single-stress-period model.  This is required if you are using `PstFrom`:


```python
df = pd.read_csv(os.path.join(t_d,"mult2model_info.csv"),index_col=0)
ifiles = set(pst.model_input_data.model_file.tolist())
#print(df.mlt_file.unique())
new_df = df.loc[df.mlt_file.apply(lambda x: pd.isna(x) or x in ifiles),:]
#new_df.shape,df.shape
#new_df.to_csv(os.path.join(t_d,"mult2model_info.csv"))
new_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>org_file</th>
      <th>model_file</th>
      <th>use_cols</th>
      <th>index_cols</th>
      <th>fmt</th>
      <th>sep</th>
      <th>head_rows</th>
      <th>upper_bound</th>
      <th>lower_bound</th>
      <th>operator</th>
      <th>mlt_file</th>
      <th>zone_file</th>
      <th>fac_file</th>
      <th>pp_file</th>
      <th>pp_fill_value</th>
      <th>pp_lower_limit</th>
      <th>pp_upper_limit</th>
      <th>zero_based</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>org\freyberg6.npf_k_layer1.txt</td>
      <td>freyberg6.npf_k_layer1.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>100</td>
      <td>0.01</td>
      <td>m</td>
      <td>mult\npfklayer1gr_inst0_grid.csv</td>
      <td>npfklayer1gr_inst0_grid.csv.zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>org\freyberg6.npf_k_layer1.txt</td>
      <td>freyberg6.npf_k_layer1.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>100</td>
      <td>0.01</td>
      <td>m</td>
      <td>mult\npfklayer1cn_inst0_constant.csv</td>
      <td>npfklayer1cn_inst0_constant.csv.zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>org\freyberg6.npf_k_layer2.txt</td>
      <td>freyberg6.npf_k_layer2.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>100</td>
      <td>0.01</td>
      <td>m</td>
      <td>mult\npfklayer2gr_inst0_grid.csv</td>
      <td>npfklayer2gr_inst0_grid.csv.zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>org\freyberg6.npf_k_layer2.txt</td>
      <td>freyberg6.npf_k_layer2.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>100</td>
      <td>0.01</td>
      <td>m</td>
      <td>mult\npfklayer2cn_inst0_constant.csv</td>
      <td>npfklayer2cn_inst0_constant.csv.zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>org\freyberg6.npf_k_layer3.txt</td>
      <td>freyberg6.npf_k_layer3.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>100</td>
      <td>0.01</td>
      <td>m</td>
      <td>mult\npfklayer3gr_inst0_grid.csv</td>
      <td>npfklayer3gr_inst0_grid.csv.zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>166</th>
      <td>org\freyberg6.sfr_packagedata.txt</td>
      <td>freyberg6.sfr_packagedata.txt</td>
      <td>[9]</td>
      <td>[0, 2, 3]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[100.0]</td>
      <td>[0.001]</td>
      <td>m</td>
      <td>mult\sfrcondcn_inst0_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>167</th>
      <td>org\freyberg6.sfr_perioddata_1.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>192</th>
      <td>org\freyberg6.ic_strt_layer1.txt</td>
      <td>freyberg6.ic_strt_layer1.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>1e+30</td>
      <td>-1e+30</td>
      <td>d</td>
      <td>NaN</td>
      <td>freyberg6.ic_strt_layer1.txt.zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>193</th>
      <td>org\freyberg6.ic_strt_layer2.txt</td>
      <td>freyberg6.ic_strt_layer2.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>1e+30</td>
      <td>-1e+30</td>
      <td>d</td>
      <td>NaN</td>
      <td>freyberg6.ic_strt_layer2.txt.zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>194</th>
      <td>org\freyberg6.ic_strt_layer3.txt</td>
      <td>freyberg6.ic_strt_layer3.txt</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>free</td>
      <td>NaN</td>
      <td>0</td>
      <td>1e+30</td>
      <td>-1e+30</td>
      <td>d</td>
      <td>NaN</td>
      <td>freyberg6.ic_strt_layer3.txt.zone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>62 rows × 18 columns</p>
</div>




```python
df.loc[:,"cycle"] = -1
```


```python
sfr = df.loc[df.model_file.str.contains("sfr_perioddata"),:].copy()
df.loc[sfr.index,"cycle"] = sfr.model_file.apply(lambda x: int(x.split("_")[-1].split(".")[0])-1)
df.loc[sfr.index.values[1:],"model_file"] = sfr.model_file.iloc[0]
df.loc[sfr.index]

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>org_file</th>
      <th>model_file</th>
      <th>use_cols</th>
      <th>index_cols</th>
      <th>fmt</th>
      <th>sep</th>
      <th>head_rows</th>
      <th>upper_bound</th>
      <th>lower_bound</th>
      <th>operator</th>
      <th>mlt_file</th>
      <th>zone_file</th>
      <th>fac_file</th>
      <th>pp_file</th>
      <th>pp_fill_value</th>
      <th>pp_lower_limit</th>
      <th>pp_upper_limit</th>
      <th>zero_based</th>
      <th>cycle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>167</th>
      <td>org\freyberg6.sfr_perioddata_1.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst0_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>168</th>
      <td>org\freyberg6.sfr_perioddata_2.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst1_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>169</th>
      <td>org\freyberg6.sfr_perioddata_3.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst2_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>170</th>
      <td>org\freyberg6.sfr_perioddata_4.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst3_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>171</th>
      <td>org\freyberg6.sfr_perioddata_5.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst4_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>4</td>
    </tr>
    <tr>
      <th>172</th>
      <td>org\freyberg6.sfr_perioddata_6.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst5_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>5</td>
    </tr>
    <tr>
      <th>173</th>
      <td>org\freyberg6.sfr_perioddata_7.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst6_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>6</td>
    </tr>
    <tr>
      <th>174</th>
      <td>org\freyberg6.sfr_perioddata_8.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst7_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>7</td>
    </tr>
    <tr>
      <th>175</th>
      <td>org\freyberg6.sfr_perioddata_9.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst8_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>8</td>
    </tr>
    <tr>
      <th>176</th>
      <td>org\freyberg6.sfr_perioddata_10.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst9_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>9</td>
    </tr>
    <tr>
      <th>177</th>
      <td>org\freyberg6.sfr_perioddata_11.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst10_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>10</td>
    </tr>
    <tr>
      <th>178</th>
      <td>org\freyberg6.sfr_perioddata_12.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst11_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>11</td>
    </tr>
    <tr>
      <th>179</th>
      <td>org\freyberg6.sfr_perioddata_13.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst12_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>12</td>
    </tr>
    <tr>
      <th>180</th>
      <td>org\freyberg6.sfr_perioddata_14.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst13_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>13</td>
    </tr>
    <tr>
      <th>181</th>
      <td>org\freyberg6.sfr_perioddata_15.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst14_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>14</td>
    </tr>
    <tr>
      <th>182</th>
      <td>org\freyberg6.sfr_perioddata_16.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst15_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>15</td>
    </tr>
    <tr>
      <th>183</th>
      <td>org\freyberg6.sfr_perioddata_17.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst16_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>16</td>
    </tr>
    <tr>
      <th>184</th>
      <td>org\freyberg6.sfr_perioddata_18.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst17_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>17</td>
    </tr>
    <tr>
      <th>185</th>
      <td>org\freyberg6.sfr_perioddata_19.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst18_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>18</td>
    </tr>
    <tr>
      <th>186</th>
      <td>org\freyberg6.sfr_perioddata_20.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst19_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>19</td>
    </tr>
    <tr>
      <th>187</th>
      <td>org\freyberg6.sfr_perioddata_21.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst20_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>20</td>
    </tr>
    <tr>
      <th>188</th>
      <td>org\freyberg6.sfr_perioddata_22.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst21_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>21</td>
    </tr>
    <tr>
      <th>189</th>
      <td>org\freyberg6.sfr_perioddata_23.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst22_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>22</td>
    </tr>
    <tr>
      <th>190</th>
      <td>org\freyberg6.sfr_perioddata_24.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst23_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>23</td>
    </tr>
    <tr>
      <th>191</th>
      <td>org\freyberg6.sfr_perioddata_25.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>[2]</td>
      <td>[0]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\sfrgr_inst24_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
rch = df.loc[df.model_file.str.contains("rch"),:]
df.loc[rch.index,"cycle"] = rch.model_file.apply(lambda x: int(x.split('_')[-1].split(".")[0])-1)
df.loc[rch.index.values[1:],"model_file"] = rch.model_file.iloc[0]
```


```python
wel = df.loc[df.model_file.str.contains("wel"),:].copy()
df.loc[wel.index,"cycle"] = wel.model_file.apply(lambda x: int(x.split('_')[-1].split(".")[0])-1)
df.loc[wel.index.values[1:],"model_file"] = wel.model_file.iloc[0]
```


```python
df.loc[wel.index]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>org_file</th>
      <th>model_file</th>
      <th>use_cols</th>
      <th>index_cols</th>
      <th>fmt</th>
      <th>sep</th>
      <th>head_rows</th>
      <th>upper_bound</th>
      <th>lower_bound</th>
      <th>operator</th>
      <th>mlt_file</th>
      <th>zone_file</th>
      <th>fac_file</th>
      <th>pp_file</th>
      <th>pp_fill_value</th>
      <th>pp_lower_limit</th>
      <th>pp_upper_limit</th>
      <th>zero_based</th>
      <th>cycle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>115</th>
      <td>org\freyberg6.wel_stress_period_data_1.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst0_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>org\freyberg6.wel_stress_period_data_1.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst0_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>117</th>
      <td>org\freyberg6.wel_stress_period_data_2.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst1_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>118</th>
      <td>org\freyberg6.wel_stress_period_data_2.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst1_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>119</th>
      <td>org\freyberg6.wel_stress_period_data_3.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst2_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>120</th>
      <td>org\freyberg6.wel_stress_period_data_3.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst2_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>121</th>
      <td>org\freyberg6.wel_stress_period_data_4.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst3_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>122</th>
      <td>org\freyberg6.wel_stress_period_data_4.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst3_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>3</td>
    </tr>
    <tr>
      <th>123</th>
      <td>org\freyberg6.wel_stress_period_data_5.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst4_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>4</td>
    </tr>
    <tr>
      <th>124</th>
      <td>org\freyberg6.wel_stress_period_data_5.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst4_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>4</td>
    </tr>
    <tr>
      <th>125</th>
      <td>org\freyberg6.wel_stress_period_data_6.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst5_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>5</td>
    </tr>
    <tr>
      <th>126</th>
      <td>org\freyberg6.wel_stress_period_data_6.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst5_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>5</td>
    </tr>
    <tr>
      <th>127</th>
      <td>org\freyberg6.wel_stress_period_data_7.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst6_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>6</td>
    </tr>
    <tr>
      <th>128</th>
      <td>org\freyberg6.wel_stress_period_data_7.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst6_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>6</td>
    </tr>
    <tr>
      <th>129</th>
      <td>org\freyberg6.wel_stress_period_data_8.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst7_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>7</td>
    </tr>
    <tr>
      <th>130</th>
      <td>org\freyberg6.wel_stress_period_data_8.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst7_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>7</td>
    </tr>
    <tr>
      <th>131</th>
      <td>org\freyberg6.wel_stress_period_data_9.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst8_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>8</td>
    </tr>
    <tr>
      <th>132</th>
      <td>org\freyberg6.wel_stress_period_data_9.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst8_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>8</td>
    </tr>
    <tr>
      <th>133</th>
      <td>org\freyberg6.wel_stress_period_data_10.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst9_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>9</td>
    </tr>
    <tr>
      <th>134</th>
      <td>org\freyberg6.wel_stress_period_data_10.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst9_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>9</td>
    </tr>
    <tr>
      <th>135</th>
      <td>org\freyberg6.wel_stress_period_data_11.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst10_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>10</td>
    </tr>
    <tr>
      <th>136</th>
      <td>org\freyberg6.wel_stress_period_data_11.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst10_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>10</td>
    </tr>
    <tr>
      <th>137</th>
      <td>org\freyberg6.wel_stress_period_data_12.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst11_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>11</td>
    </tr>
    <tr>
      <th>138</th>
      <td>org\freyberg6.wel_stress_period_data_12.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst11_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>11</td>
    </tr>
    <tr>
      <th>139</th>
      <td>org\freyberg6.wel_stress_period_data_13.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst12_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>12</td>
    </tr>
    <tr>
      <th>140</th>
      <td>org\freyberg6.wel_stress_period_data_13.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst12_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>12</td>
    </tr>
    <tr>
      <th>141</th>
      <td>org\freyberg6.wel_stress_period_data_14.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst13_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>13</td>
    </tr>
    <tr>
      <th>142</th>
      <td>org\freyberg6.wel_stress_period_data_14.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst13_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>13</td>
    </tr>
    <tr>
      <th>143</th>
      <td>org\freyberg6.wel_stress_period_data_15.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst14_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>14</td>
    </tr>
    <tr>
      <th>144</th>
      <td>org\freyberg6.wel_stress_period_data_15.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst14_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>14</td>
    </tr>
    <tr>
      <th>145</th>
      <td>org\freyberg6.wel_stress_period_data_16.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst15_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>15</td>
    </tr>
    <tr>
      <th>146</th>
      <td>org\freyberg6.wel_stress_period_data_16.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst15_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>15</td>
    </tr>
    <tr>
      <th>147</th>
      <td>org\freyberg6.wel_stress_period_data_17.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst16_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>16</td>
    </tr>
    <tr>
      <th>148</th>
      <td>org\freyberg6.wel_stress_period_data_17.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst16_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>16</td>
    </tr>
    <tr>
      <th>149</th>
      <td>org\freyberg6.wel_stress_period_data_18.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst17_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>17</td>
    </tr>
    <tr>
      <th>150</th>
      <td>org\freyberg6.wel_stress_period_data_18.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst17_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>17</td>
    </tr>
    <tr>
      <th>151</th>
      <td>org\freyberg6.wel_stress_period_data_19.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst18_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>18</td>
    </tr>
    <tr>
      <th>152</th>
      <td>org\freyberg6.wel_stress_period_data_19.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst18_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>18</td>
    </tr>
    <tr>
      <th>153</th>
      <td>org\freyberg6.wel_stress_period_data_20.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst19_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>19</td>
    </tr>
    <tr>
      <th>154</th>
      <td>org\freyberg6.wel_stress_period_data_20.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst19_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>19</td>
    </tr>
    <tr>
      <th>155</th>
      <td>org\freyberg6.wel_stress_period_data_21.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst20_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>20</td>
    </tr>
    <tr>
      <th>156</th>
      <td>org\freyberg6.wel_stress_period_data_21.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst20_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>20</td>
    </tr>
    <tr>
      <th>157</th>
      <td>org\freyberg6.wel_stress_period_data_22.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst21_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>21</td>
    </tr>
    <tr>
      <th>158</th>
      <td>org\freyberg6.wel_stress_period_data_22.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst21_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>21</td>
    </tr>
    <tr>
      <th>159</th>
      <td>org\freyberg6.wel_stress_period_data_23.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst22_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>22</td>
    </tr>
    <tr>
      <th>160</th>
      <td>org\freyberg6.wel_stress_period_data_23.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst22_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>22</td>
    </tr>
    <tr>
      <th>161</th>
      <td>org\freyberg6.wel_stress_period_data_24.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst23_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>23</td>
    </tr>
    <tr>
      <th>162</th>
      <td>org\freyberg6.wel_stress_period_data_24.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst23_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>23</td>
    </tr>
    <tr>
      <th>163</th>
      <td>org\freyberg6.wel_stress_period_data_25.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welcst_inst24_constant.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>24</td>
    </tr>
    <tr>
      <th>164</th>
      <td>org\freyberg6.wel_stress_period_data_25.txt</td>
      <td>freyberg6.wel_stress_period_data_1.txt</td>
      <td>[3]</td>
      <td>[0, 1, 2]</td>
      <td>free</td>
      <td></td>
      <td>0</td>
      <td>[1e+30]</td>
      <td>[-1e+30]</td>
      <td>m</td>
      <td>mult\welgrd_inst24_grid.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[df.cycle!=-1,["org_file","model_file","cycle"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>org_file</th>
      <th>model_file</th>
      <th>cycle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>org\freyberg6.rch_recharge_1.txt</td>
      <td>freyberg6.rch_recharge_1.txt</td>
      <td>0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>org\freyberg6.rch_recharge_1.txt</td>
      <td>freyberg6.rch_recharge_1.txt</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>org\freyberg6.rch_recharge_2.txt</td>
      <td>freyberg6.rch_recharge_1.txt</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>org\freyberg6.rch_recharge_2.txt</td>
      <td>freyberg6.rch_recharge_1.txt</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40</th>
      <td>org\freyberg6.rch_recharge_3.txt</td>
      <td>freyberg6.rch_recharge_1.txt</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>187</th>
      <td>org\freyberg6.sfr_perioddata_21.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>20</td>
    </tr>
    <tr>
      <th>188</th>
      <td>org\freyberg6.sfr_perioddata_22.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>21</td>
    </tr>
    <tr>
      <th>189</th>
      <td>org\freyberg6.sfr_perioddata_23.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>22</td>
    </tr>
    <tr>
      <th>190</th>
      <td>org\freyberg6.sfr_perioddata_24.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>23</td>
    </tr>
    <tr>
      <th>191</th>
      <td>org\freyberg6.sfr_perioddata_25.txt</td>
      <td>freyberg6.sfr_perioddata_1.txt</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 3 columns</p>
</div>




```python
df.to_csv(os.path.join(t_d,"mult2model_info.global.csv"))
```


```python
shutil.copy2("prep_mult.py",os.path.join(t_d,"prep_mult.py"))
```




    'freyberg6_da_template\\prep_mult.py'




```python
lines = open(os.path.join(t_d,"forward_run.py"),'r').readlines()
with open(os.path.join(t_d,"forward_run.py"),'w') as f:
    for line in lines:
        if "apply_list_and_array_pars" in line:
            f.write("    pyemu.os_utils.run('python prep_mult.py')\n")
        f.write(line)
```

# OMG that was brutal


```python
pst.pestpp_options.pop("forecasts",None)
pst.control_data.noptmax = 0
pst.write(os.path.join(t_d,"freyberg_mf6.pst"),version=2)
pyemu.os_utils.run("pestpp-da freyberg_mf6.pst",cwd=t_d)
```

    noptmax:0, npar_adj:29653, nnz_obs:5
    

Wow, that takes a lot longer...this is the price of sequential estimation...


```python

```
