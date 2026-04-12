clear; clc;

% % % % % baseDir = "data/usable";
% % % % % 
% % % % % % quick check
% % % % % folders = [
% % % % % 	fullfile(baseDir,"P1","SAFE")
% % % % % 	fullfile(baseDir,"P1","NOMINAL")
% % % % % 	fullfile(baseDir,"P2","SAFE")
% % % % % 	fullfile(baseDir,"P2","NOMINAL")
% % % % % ];
% % % % % 
% % % % % for i = 1:size(folders,1)
% % % % % 	if ~isfolder(folders(i))
% % % % % 		fprintf("WARN: missing folder: %s\n", folders(i));
% % % % % 	else
% % % % % 		fprintf("OK:   %s\n", folders(i));
% % % % % 	end
% % % % % end
% % % % % 

baseDir = "data/usable";

phases = ["P1","P2"];
modes  = ["SAFE","NOMINAL"];

noisePctLevels = [0, 0.5, 1, 2, 3];
regions = ["aposelene", "leaving_aposelene", "approaching_aposelene", "periselene"];

% We accumulate per cell using "pooled" stats:
% keep:
%   Ntot, sumX, sumX2  (to compute global mean/std across all episodes)
% where episodes are the MonteCarlo simulations inside each .mat.
%
% We do this by reconstructing:
%   sumX  += n_i * mu_i
%   sumX2 += (n_i-1)*sigma_i^2 + n_i*mu_i^2
%
% This is exact if mu_i/sigma_i are computed over the n_i episodes
% (which they are in your MonteCarloInfo).
tofAgg = containers.Map(); % key -> struct('N',...,'sumX',...,'sumX2',...)
dvAgg  = containers.Map();

keyOf = @(phase,reg,mode,np) sprintf("%s|%s|%s|%g", phase, reg, mode, np);

for p = 1:numel(phases)
	phaseName = phases(p);

	for m = 1:numel(modes)
		modeName = modes(m);

		folder = fullfile(baseDir, phaseName, modeName);
		if ~isfolder(folder)
			fprintf("WARN: folder not found: %s\n", folder);
			continue;
		end

		files = dir(fullfile(folder, "*.mat"));
		fprintf("%s: %d files\n", folder, numel(files));

		for k = 1:numel(files)
			fpath = fullfile(files(k).folder, files(k).name);

			meta = parse_mc_filename(files(k).name);
			if ~meta.validRegion
				continue;
			end

			noisePct = meta.noisePct;
			if ~any(abs(noisePctLevels - noisePct) < 1e-9)
				continue;
			end

			% ---- load + run your analysis ----
			data = load_mc_data_struct(fpath);

			% MonteCarloInfo prints a lot; if you want silence:
			% MC_results = suppress_output(@() MonteCarloInfo(data));
			% But easiest is just call it directly:
			[~,~,MC_results] = MonteCarloInfo(data);

			% n episodes in this file
			n_i = double(data.n_population);

			% ---- TOF [s] from MC_results ----
			tof_mu_s = double(MC_results.meanTOF)  * double(data.param.tc);
			tof_sd_s = double(MC_results.sigmaTOF) * double(data.param.tc);

			% ---- DeltaV [m/s] from MC_results (YOUR definition) ----
			dt = (1/double(data.param.freqGNC)) * double(data.param.tc);
			dv_mu = double(MC_results.meanMass)  / dt;
			dv_sd = double(MC_results.sigmaMass) / dt;

			K = keyOf(phaseName, meta.region, modeName, noisePct);

			tofAgg = add_pooled_stats(tofAgg, K, n_i, tof_mu_s, tof_sd_s);
			dvAgg  = add_pooled_stats(dvAgg,  K, n_i, dv_mu,    dv_sd);
		end
	end
end

% Build long tables from aggregators
T_TOF = build_long_from_agg(tofAgg, phases, regions, modes, noisePctLevels, "TOF", "s");
T_DV  = build_long_from_agg(dvAgg,  phases, regions, modes, noisePctLevels, "DeltaV", "m/s");

% Wide mean matrices
W_TOF_safe    = make_wide(T_TOF, "SAFE",    "MeanValue");
W_TOF_nominal = make_wide(T_TOF, "NOMINAL", "MeanValue");
W_DV_safe     = make_wide(T_DV,  "SAFE",    "MeanValue");
W_DV_nominal  = make_wide(T_DV,  "NOMINAL", "MeanValue");

% =========================================================================
% Aggregation helpers (pooled)
% =========================================================================
function aggMap = add_pooled_stats(aggMap, key, n_i, mu_i, sd_i)
	if n_i <= 0 || isnan(mu_i) || isnan(sd_i)
		return;
	end

	% Convert (mu, sd, n) -> sumX, sumX2
	% sample variance: sd^2 = (1/(n-1)) * sum((x-mu)^2)
	% so: sumX2 = sum(x^2) = (n-1)*sd^2 + n*mu^2
	sumX_i  = n_i * mu_i;
	sumX2_i = (max(n_i-1,0)) * (sd_i^2) + n_i * (mu_i^2);

	if isKey(aggMap, key)
		A = aggMap(key);
		A.N     = A.N     + n_i;
		A.sumX  = A.sumX  + sumX_i;
		A.sumX2 = A.sumX2 + sumX2_i;
		aggMap(key) = A;
	else
		A = struct("N", n_i, "sumX", sumX_i, "sumX2", sumX2_i);
		aggMap(key) = A;
	end
end

function [mu, sd, N] = finalize_pooled(A)
	N = double(A.N);
	if N <= 0
		mu = NaN; sd = NaN; return;
	end

	mu = A.sumX / N;

	if N == 1
		sd = 0;
		return;
	end

	% sample variance: (sum(x^2) - N*mu^2) / (N-1)
	var = (A.sumX2 - N * mu^2) / (N - 1);
	var = max(var, 0); % numerical safety
	sd = sqrt(var);
end

% =========================================================================
% Build tables
% =========================================================================
function T = build_long_from_agg(aggMap, phases, regions, modes, noisePctLevels, metricName, unitStr)
	rows = {};
	keyOf = @(phase,reg,mode,np) sprintf("%s|%s|%s|%g", phase, reg, mode, np);

	for p = 1:numel(phases)
		for r = 1:numel(regions)
			for m = 1:numel(modes)
				for n = 1:numel(noisePctLevels)
					K = keyOf(phases(p), regions(r), modes(m), noisePctLevels(n));

					if ~isKey(aggMap, K)
						mu = NaN; sd = NaN; NN = 0; vs = "";
					else
						A = aggMap(K);
						[mu, sd, NN] = finalize_pooled(A);
						vs = sprintf("%.3f ± %.3f %s", mu, sd, unitStr);
					end

					rows(end+1,:) = {phases(p), regions(r), modes(m), noisePctLevels(n), mu, sd, NN, vs}; %#ok<AGROW>
				end
			end
		end
	end

	T = cell2table(rows, "VariableNames", ["Phase","Region","Mode","NoisePct","MeanValue","StdValue","N","ValueStr"]);
	% keep as strings to avoid ordinal categorical issues
	T.Phase  = string(T.Phase);
	T.Region = string(T.Region);
	T.Mode   = string(T.Mode);

	T.Properties.Description = sprintf("%s table (%s)", metricName, unitStr);
end

function W = make_wide(T, modeName, valueField)
	S = T(T.Mode == string(modeName), :);
	S.RowKey = string(S.Phase) + " | " + string(S.Region);
	W = unstack(S, valueField, "NoisePct", "AggregationFunction", @mean);
	W = movevars(W, "RowKey", "Before", 1);
end

% =========================================================================
% Filename parsing (your naming convention)
% =========================================================================
function meta = parse_mc_filename(fname)
	meta = struct();
	meta.validRegion = false;
	meta.region = "";
	meta.noisePct = 0;

	s = lower(string(fname));

	% Noise token: "..._N0.005_..." -> noisePct = N*100
	tok = regexp(s, "_n([0-9]*\.?[0-9]+)_", "tokens", "once");
	if isempty(tok)
		noiseVal = 0;
	else
		noiseVal = str2double(tok{1});
		if isnan(noiseVal), noiseVal = 0; end
	end

	noisePct = noiseVal * 100; % N0.005 -> 0.5%
	levels = [0, 0.5, 1, 2, 3];
	[~,ix] = min(abs(levels - noisePct));
	if abs(levels(ix) - noisePct) < 1e-6
		noisePct = levels(ix);
	end
	meta.noisePct = noisePct;

	% Region tokens
	if contains(s, "approaching_aposelene")
		meta.region = "approaching_aposelene"; meta.validRegion = true; return;
	end
	if contains(s, "leaving_aposelene")
		meta.region = "leaving_aposelene";     meta.validRegion = true; return;
	end
	if contains(s, "periselene")
		meta.region = "periselene";            meta.validRegion = true; return;
	end
	if contains(s, "aposelene")
		meta.region = "aposelene";             meta.validRegion = true; return;
	end
end

% =========================================================================
% Load .mat -> get "data" struct
% =========================================================================
function data = load_mc_data_struct(fpath)
	S = load(fpath);

	if isfield(S, "data")
		data = S.data;
		return;
	end

	fns = fieldnames(S);
	if numel(fns) == 1
		data = S.(fns{1});
		if ~isstruct(data)
			error("File %s has one variable but it's not a struct.", fpath);
		end
		return;
	end

	candidates = ["MC","mc","MonteCarlo","montecarlo","out","results"];
	for i = 1:numel(candidates)
		if isfield(S, candidates(i))
			data = S.(candidates(i));
			if isstruct(data), return; end
		end
	end

	error("Cannot find MonteCarlo struct in %s (fields: %s).", fpath, strjoin(string(fns), ", "));
end

% =========================================================================
% Optional: suppress MonteCarloInfo prints (if you want silence)
% =========================================================================
function out = suppress_output(funHandle)
	tmp = evalc('out = funHandle();');
end