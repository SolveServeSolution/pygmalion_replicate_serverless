#!/usr/bin/python3 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSeq2SeqLM,
)
import torch
import os
from dotenv import load_dotenv
from cog import BasePredictor, Input, Path
load_dotenv()

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        # --- gpu detection ---
        self.device = torch.device("cpu")  # default to cpu
        self.use_gpu = torch.cuda.is_available()
        print("Detecting GPU...")
        if self.use_gpu:
            print("GPU detected!")
            self.device = torch.device("cuda")
            print("Using GPU...")
        else:
            print("GPU not detected, using CPU...")
        # ---------- load Conversation model ----------
        print("Initilizing model....")
        print("Loading language model...")

        # --- for testing, small model ---
        if os.environ.get("DEV") != None: #
            print("Loading model for testing...")
            model_name = "PygmalionAI/pygmalion-1.3b"

        else: # --- for production, big model ---
            print("Loading model for production...")
            model_name = "PygmalionAI/pygmalion-2-13b"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        config = AutoConfig.from_pretrained(model_name, is_decoder=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
        )

        if self.use_gpu:  # load model to GPU
            print("Loading model at full precision...")
            self.model = self.model.to(self.device)
            # half precision
            # model = model.half()

    # The arguments and types the model takes as input
    def predict(self,
            prompt: str = Input(description="what to put into LLM"),
    ) -> Path:
        """Run a single prediction on the model"""
        # --- input ---
            
        print("Proceeding... with pure english conversation")
        ## ----------- Will move this to server later -------- (16GB ram needed at least)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.use_gpu:
            inputs = inputs.to(self.device)
        out = self.model.generate(
            **inputs,
            max_length=len(inputs["input_ids"][0]) + 150,  # todo 200 ?
            pad_token_id=self.tokenizer.eos_token_id,
        )
        conversation = self.tokenizer.decode(out[0])
        return conversation