from model import ESMEvaluator
from LigandMPNN.package import MPNNModel
import argparse
import os
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process multiple PDB files with AF3 based optimizer')
    parser.add_argument('--pdb_list', type=str, required=False,
                       help='Path to text file containing list of PDB files')
    parser.add_argument('--input_file', type=str, required=False,
                       help='input file path')
    parser.add_argument('--fix_res_index', type=str, required=False, 
                        help='Fixed residue indices, e.g. A1 B4 but be careful, we will reindex all to begin with 1')
    parser.add_argument('--fix_chain_index', type=str, required=False,
                        help='Fixed chain indices, e.g. A B')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for output files')
    parser.add_argument('--num_seqs', type=int, default=8,
                       help='Number of proteinMPNN or LigadnMPNN+proteinMPNN seqs to perform self consistency')
    parser.add_argument('--esmhead', action='store_true', default=False,
                    help='whether to use esmhead to validate sequence quality')
    parser.add_argument('--mpnn',  type=str,  required=False,
                    help='which mpnn model do you choose proteinmpnn ligandmpnn ligandmpnn_plus_proteinmpnn')
    parser.add_argument('--mpnn_temperature', type=float, default=0.1,
                       help='mpnn temperature to use')
    return parser.parse_args()

def run_purempnn_evaluation(mpnn_model,scaffold_path,mpnn_config_dict,pocket_res_to_fix,weights_str,output_dir,bias_AA,symmetry_residues):  
    

    print(f"{mpnn_config_dict['model_name']} design")
    output_dir_mpnn =os.path.join(output_dir,f"{mpnn_config_dict['model_name']}")
    sequences_stack,pdb_path_stack=mpnn_model.single_protein_mpnn_design(scaffold_path=scaffold_path, 
                                      output_dir_mpnn=output_dir_mpnn, 
                                      numbers_seqs=mpnn_config_dict['num_seqs'],   
                                      chains_to_design="",
                                      fixed_res=pocket_res_to_fix, 
                                      redesigned_residues="", 
                                      symmetry_residues=symmetry_residues,
                                      input_bias_AA = bias_AA,
                                      weights_str=weights_str,)

    return  sequences_stack,pdb_path_stack

import time
def run_selection_process(sequences, packed_paths, evaluator, scaffold_path, num_seqs,batch_size):
    """
    Performs multi-round tournament-style selection based on prediction scores.

    Args:
        sequences (list): List of generated amino acid sequences.
        packed_paths (list): List of PDB file paths corresponding to the sequences.
        evaluator (object): Evaluator object used for predicting scores.
        scaffold_path (str): Path to the scaffold PDB file.
        mpnn_config_dict (dict): Dictionary containing configuration, including 'num_seqs'.

    Returns:
        list: Filtered list of (sequence, packed_path, score) tuples, sorted in descending order by score.
    """
    current_data = [(seq, packed) for seq, packed in zip(sequences, packed_paths)]
    #print(current_data)
    # target number of sequences to retain
    target_num_seqs = num_seqs # extract from mpnn_config_dict
    start_time = time.time()
    print(f"init counts: {len(current_data)}")
    print(f"target counts: {target_num_seqs}")
    # make sure target_num_seqs at least 1
    if target_num_seqs == 0:
        target_num_seqs = 1 
    while len(current_data) > target_num_seqs:
        batch_size = 4 # max number of sequences to process in one batch
        all_scores_for_round = []
        
        # in batch to predict
        for i in range(0, len(current_data), batch_size):
            batch_sequences = [item[0] for item in current_data[i : i + batch_size]]
            batch_packed_paths = [item[1] for item in current_data[i : i + batch_size]]

            batch_scores = evaluator.predict(batch_sequences, scaffold_path)

            for j, (seq, packed) in enumerate(zip(batch_sequences, batch_packed_paths)):
                all_scores_for_round.append((seq, packed, batch_scores[j]))
        
        # Sort all sequences in the current round by score (descending)
        all_scores_for_round.sort(key=lambda x: -x[2])  # x[2] is the score
        
        # Select top half sequences for the next round
        # For odd numbers, take (N+1)/2 to ensure at least half are kept
        next_round_count = max(target_num_seqs, (len(all_scores_for_round) + 1) // 2)
        
        # Ensure we don't select more sequences than available
        # The loop condition already prevents current round from being smaller than target
        
        current_data = [(item[0], item[1]) for item in all_scores_for_round[:next_round_count]]
        
        print(f"finial count: {len(current_data)}")
        if len(current_data) == 0:  # Prevent infinite loops and handle empty lists
            print("Warning: Filtered sequence list is empty, terminating early.")
            break
    
        # Final results need to follow the complete interaction_data format: (seq, packed, 0, 0, 0, score)
        # Here 'score' should be the final evaluation score, other fields are filled with 0
        final_results = []
        for seq, packed in current_data:
            # Find the final score for this sequence
            # In theory, all_scores_for_round stores the scores from the final round
            # To maintain compatibility with original_length_sequences format, we need to populate the score field
            # If evaluator.predict returns the final score, it should be used here

            # Look up the final score for this sequence (this requires tracking scores across rounds)
            # For simplicity, we assume you need both final sequences and their corresponding scores
            # We'll search for the score in the final round's results (all_scores_for_round)

            # Find the current sequence's score from the final round
            final_score = 0
            for item in all_scores_for_round:  # Ensure all_scores_for_round contains complete final round data
            if item[0] == seq and item[1] == packed:
                final_score = item[2]
                break
        
        final_results.append((seq, packed, 0, 0, 0, final_score))

    final_results.sort(key=lambda x: -x[5])

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return final_results



def main():
    args = parse_arguments()
    mpnn_config_dict = {
            "temperature": args.mpnn_temperature,
            "model_name": args.mpnn, #"ligandmpnn_plus_proteinmpnn"
            "num_seqs": 1
            }
    if args.esmhead:
        checkpoints_to_run = os.path.dirname(os.path.abspath(__file__))+ "/ckpt/epoch_1_without_esm2.pth"
        ##! need change
        esm_name = "facebook/esm2_t33_650M_UR50D"
        evaluator = ESMEvaluator(checkpoints_to_run,esm_name)
        mpnn_config_dict["num_seqs"] = args.num_seqs
        mpnn_config_dict["num_seqs"] = mpnn_config_dict["num_seqs"] * 8
    else:
        evaluator = None

    if args.pdb_list:
        with open(args.pdb_list, 'r') as f:
            pdb_files = [line.strip() for line in f if line.strip()]
    elif args.input_file:
        pdb_files = [args.input_file]
    else:
        raise ValueError("You must specify either --pdb_list or --input_file")
    
    if args.mpnn == "protein_mpnn":
        mpnn_model = MPNNModel(model_name="protein_mpnn",
                        T=args.mpnn_temperature, 
                        ligand_mpnn_use_side_chain_context=1,
                        ligand_mpnn_use_atom_context=1,
                        number_of_packs_per_design=1,
                        pack_side_chains=1,
                        parse_atoms_with_zero_occupancy=1,
                        pack_with_ligand_context= 0,
                        repack_everything=1)
        #raise ValueError("protein_mpnn can not use in small molecular or dna or rna system design")
    if args.mpnn == "ligand_mpnn":
        mpnn_model = MPNNModel(model_name=args.mpnn,
                        T=args.mpnn_temperature, 
                        ligand_mpnn_use_side_chain_context=1,
                        ligand_mpnn_use_atom_context=1,
                        number_of_packs_per_design=1,
                        pack_side_chains=1,
                        parse_atoms_with_zero_occupancy=1,
                        pack_with_ligand_context= 0,
                        repack_everything=1)
    
    
    for pdb_file in pdb_files:
        sequences,packed_paths=run_purempnn_evaluation(mpnn_model,pdb_file,mpnn_config_dict,"","",args.output_dir,"","")
        print(sequences,packed_paths)
        if evaluator :
            num_seqs = int(mpnn_config_dict["num_seqs"]/8 )
            #scores = evaluator.predict(sequences, scaffold_path)
            #interaction_data = [(seq, packed, 0, 0, 0, score) for seq, packed, score in zip(sequences, packed_paths, scores)]
            #original_length_sequences = sorted(interaction_data, key=lambda x: -x[5])[:num_seqs]
            batchsizes = [2, 4, 8]
            last_error = None

            for batchsize in batchsizes:
                try:
                    original_length_sequences = run_selection_process(
                        sequences, packed_paths, evaluator, pdb_file, num_seqs, batchsize,
                    )
                    print(f"✅ Successfully ran with batchsize: {batchsize}")
                    break  # successful execution, exit the loop
                except Exception as e:
                    last_error = e
                    print(f"⚠️ Batchsize {batchsize} failed: {e}. Trying smaller batchsize...")
            else:  # loop exhausted without break
                print("❌ All batchsizes failed. Raising last error.")
                raise last_error
        else:
            print("no esmhead")
        print(original_length_sequences)


if __name__ == "__main__":
    main()
