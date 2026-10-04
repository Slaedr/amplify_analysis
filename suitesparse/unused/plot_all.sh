
python plot_spmv_all_times.py --csr-dir results_spmv_ampcsr --ell-dir results_spmv_ampell
python plot_spmv_vs_amp.py --results-dir results_spmv_ampcsr --base-format csr
python plot_spmv_vs_amp.py --results-dir results_spmv_ampell --base-format ell
python plot_spmv_error_vs_amp.py --results-dir results_spmv_ampcsr --base-format csr
python plot_spmv_error_vs_amp.py --results-dir results_spmv_ampell --base-format ell
python plot_spmv_amp_bins.py --results-dir results_spmv_ampcsr
python plot_spmv_amp_bins.py --results-dir results_spmv_ampell
python plot_spmv_amp_storage.py --csr-dir results_spmv_ampcsr --ell-dir results_spmv_ampell
