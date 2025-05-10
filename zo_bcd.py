    def zo_perturb_parameters_lora_bcd(self, random_seed=None, scaling_factor=1, total_steps = None):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)

        for name, param in self.named_parameters_to_optim:
            grad_sparsity = self.get_grad_sparsity_by_name(name)
            # z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            z = 0
            if total_steps % 220 <= 200:
                if "lora_B" in name:  
                    z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            else:
                if "lora_A" in name:  
                    z = torch.normal(mean=0, std=0.5, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                
            # if "lora_B" in name:  
            #     z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            # else:
            #     z = torch.normal(mean=0, std=0.1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            if grad_sparsity is not None:
                z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0
            param.data = param.data + scaling_factor * z * self.args.zo_eps 
    
    @torch.no_grad()
    def zo_step_bcd(self, model, inputs, total_steps):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        # NOTE: when sparse_grad is set to True, it will also check the args.gradient_sparsity,
        # so it does not necessarily use sparse grad.
        self.zo_perturb_parameters_lora_bcd(scaling_factor=1, total_steps=total_steps)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        assert args.q == 1, "only support q=1 for the memory efficiency. If you want to implement q>1, need to store random seeds to save memory. In addition, we need to set different random seed for different z in the q-loop."
        for _ in range(args.q):  # TODO shall we change the seed?
            if self.args.perturbation_mode == "one_side":
                self.zo_perturb_parameters_lora_bcd(scaling_factor=-1)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / self.args.zo_eps).item()
            else:  # two side perturbation
                self.zo_perturb_parameters_lora_bcd(scaling_factor=-2, total_steps=total_steps)
                loss2 = self.zo_forward(model, inputs)
                self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
                print(self.projected_grad)

                # Reset model back to its parameters at start of step
                self.zo_perturb_parameters_lora_bcd(scaling_factor=1, total_steps=total_steps)

            # Set the random seed to ensure that we sample the same z for perturbation/update
            torch.manual_seed(self.zo_random_seed)
            self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = None
                if total_steps % 220 <= 200:
                    if "lora_B" in name:  
                        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                else:
                    if "lora_A" in name:  
                        z = torch.normal(mean=0, std=0.5, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                # z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                #                  dtype=param.data.dtype)
                grad_sparsity = self.get_grad_sparsity_by_name(name)
                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

                
                if z is not None:
                    if args.trainer == "zo_sign_opt":
                        # ----signOpt_orig
                        # TODo why do we multiply lr here? We will multiply lr twice?
                        # graddiff_times_z = np.sign(self.projected_grad) * z                    
                            graddiff_times_z = torch.sign(self.projected_grad * z)
                        # ----signOpt_mul_sign
                        # graddiff_times_z = self._get_learning_rate() * torch.sign(self.projected_grad * z)
                    else:
                        # ----mezo original
                        graddiff_times_z = self.projected_grad * z
                
                if total_steps % 220 <= 200:
                    if "lora_B" in name: 
                        param.grad = graddiff_times_z / args.q
                        self.optimizer.step() 
                        param.grad = None
                else:
                    if "lora_A" in name:
                        param.grad = graddiff_times_z / args.q
                        for group in self.optimizer.param_groups:
                            group['lr'] = group['lr'] / 2
                        self.optimizer.step() 
                        for group in self.optimizer.param_groups:
                            group['lr'] = group['lr'] * 2
                        param.grad = None
                    

        assert self.args.gradient_accumulation_steps == 1

        return loss1