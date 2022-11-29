from models.PCP_model import build
from models.saliency import build_saliency
import argparse

def build_model(args, training=False):
    return build(args,training)

def saliency_model():
    return build_saliency()