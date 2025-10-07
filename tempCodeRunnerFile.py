from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, CrossEncoder, util, models
from collections import defaultdict, Counter
import pandas as pd
import torch
import math
import re
import numpy as np
import os, json
import torch.nn as nn