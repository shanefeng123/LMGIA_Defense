import random
import torch


class BeamSearchOptimizer:
    def __init__(self, model, hidden_states, batch, device, neighbours, tokenizer, beam_size=3, iteration=10, alpha=1,
                 beta=10):
        self.model = model
        self.hidden_states = hidden_states
        self.batch = batch
        self.device = device
        self.neighbours = neighbours
        self.valid_neighbours = []
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.initialization = []
        self.beams = []
        self.beams_objs = []
        self.iteration = iteration
        self.alpha = alpha
        self.beta = beta
        self.max_output_mse = None
        self.max_embedding_mse = None
        self.min_output_mse = None
        self.min_embedding_mse = None

    def initialize(self):
        batch_tokens = self.batch["input_ids"].tolist()
        for i in range(len(batch_tokens)):
            tokens = batch_tokens[i]
            for token in tokens:
                if token in self.tokenizer.all_special_ids:
                    self.valid_neighbours.append(None)
                else:
                    self.valid_neighbours.append(self.neighbours[token])

        self.max_output_mse, self.max_embedding_mse, self.min_output_mse, self.min_embedding_mse = self.get_min_max_mse()
        for i in range(len(batch_tokens)):
            tokens = batch_tokens[i]
            dummy_tokens = []
            for token in tokens:
                if token in self.tokenizer.all_special_ids:
                    dummy_tokens.append(token)
                else:
                    dummy_tokens.append(random.choice(self.neighbours[token]))
            self.initialization.append(dummy_tokens)
        self.beams.append(self.initialization)
        for beam in self.beams:
            self.beams_objs.append(self.objective_function(beam))

    def get_min_max_mse(self):
        batch_tokens = self.batch["input_ids"].tolist()
        dummy_tokens = []
        for i in range(len(batch_tokens)):
            tokens = batch_tokens[i]
            for token in tokens:
                if token in self.tokenizer.all_special_ids:
                    dummy_tokens.append(token)
                else:
                    dummy_tokens.append(self.neighbours[token][0])
        min_output_mse, min_embed_mse = self.get_mse([dummy_tokens])

        dummy_tokens = []
        for i in range(len(batch_tokens)):
            tokens = batch_tokens[i]
            dummy_tokens = []
            for token in tokens:
                if token in self.tokenizer.all_special_ids:
                    dummy_tokens.append(token)
                else:
                    dummy_tokens.append(self.neighbours[token][-1])
        max_output_mse, max_embed_mse = self.get_mse([dummy_tokens])
        return max_output_mse, max_embed_mse, min_output_mse, min_embed_mse

    def get_mse(self, solution):
        with torch.no_grad():
            solution_output = self.model(input_ids=torch.tensor(solution).to(self.device),
                                         attention_mask=self.batch["attention_mask"].to(self.device),
                                         output_hidden_states=True)
        solution_hidden_states = solution_output.hidden_states
        solution_embedding = self.model.embeddings.word_embeddings(torch.tensor(solution).to(self.device))
        batch_embedding = self.model.embeddings.word_embeddings(self.batch["input_ids"].to(self.device))
        # output mse loss is the MSE between the hidden states
        output_mse = 0
        for j in range(len(self.hidden_states)):
            output_mse += torch.nn.functional.mse_loss(self.hidden_states[j], solution_hidden_states[j])
        # embedding mse loss is the MSE between the embeddings excluding [CLS], [SEP], and [PAD] tokens
        embed_mse = torch.nn.functional.mse_loss(
            solution_embedding[self.batch["attention_mask"] == 1],
            batch_embedding[self.batch["attention_mask"] == 1])
        return output_mse.item(), embed_mse.item()

    def objective_function(self, solution):
        output_mse, embed_mse = self.get_mse(solution)
        # min max normalization
        output_mse = (output_mse - self.min_output_mse) / (self.max_output_mse - self.min_output_mse)
        embed_mse = (embed_mse - self.min_embedding_mse) / (self.max_embedding_mse - self.min_embedding_mse)
        loss = self.alpha * output_mse - self.beta * embed_mse
        return loss

    def optimize(self):
        # for i in range(self.iteration):
        i = 0
        while True:
            print("Optimization iter", i + 1)
            current_best_solution = self.beams[0]
            solutions = []
            objs = []
            for j in range(len(self.beams)):
                beam = self.beams[j]
                for k in range(len(beam)):
                    sequence = beam[k]
                    for l in range(len(sequence)):
                        if self.valid_neighbours[l] is None:
                            continue
                        else:
                            for neighbour in self.valid_neighbours[l]:
                                new_sequence = sequence.copy()
                                new_sequence[l] = neighbour
                                new_solution = beam.copy()
                                new_solution[k] = new_sequence
                                solutions.append(new_solution)
                                objs.append(self.objective_function(new_solution))
            solutions.extend(self.beams)
            objs.extend(self.beams_objs)
            combined = list(zip(solutions, objs))
            combined.sort(key=lambda x: x[1])
            self.beams = [x[0] for x in combined[:self.beam_size]]
            self.beams_objs = [x[1] for x in combined[:self.beam_size]]
            print("Best loss function value:", self.beams_objs[0])
            i += 1
            if current_best_solution == self.beams[0]:
                print("Stuck in local minimum")
                break

        return self.beams[0], self.objective_function(self.beams[0])
