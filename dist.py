import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import time
import altair as alt

st.title("Modélisation de la distribution")

length = 30000
bins=500

st.sidebar.markdown("# Données d'entrée")

dist = st.sidebar.radio("Choix de la distribution", ("Gamma", "Normal", "Lognormal"))

st.sidebar.markdown("# Paramètres")

perturb = st.sidebar.slider("Bruit Gaussien", 0.0, 1.0, 0.2, 0.05)
sse_thr = st.sidebar.slider("Tolérance", 0.0, 0.5, 0.05, 0.01)
max_time = st.sidebar.slider("Temps Max.", 0, 30, 10, 1)

dist_names = ['norm', 'beta','gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2']

PERTURB_FACTOR = 20

def plot_mle(data):

	with st.spinner("Calcul du maximum de vraisemblance..."):

		# Histogramme des données
		y, x = np.histogram(data, bins=bins, density=True)
		# Milieu de chaque classe
		x = (x + np.roll(x, -1))[:-1] / 2.0
		y = y + np.random.randn(bins)*perturb/PERTURB_FACTOR

		time_base = time.time()
		timeout = time_base + max_time

		# Initialize the SSE
		sse = np.inf
		to_break = False

		# For each distribution
		for name in dist_names:

			# Break at timeout
			if time.time() > timeout:
				st.error("Temps maximal écoulé.")
				to_break = True
				break

			# Get the parameters and fit the distribution
			dist = getattr(scipy.stats, name)
			param = dist.fit(data)

			# Parameters of the fitted model
			loc = param[-2]
			scale = param[-1]
			arg = param[:-2]

			# Generate the PDF and compute the SSE
			pdf = dist.pdf(x, *arg, loc=loc, scale=scale)
			model_sse = np.sum((y - pdf)**2)

			# If the SSE is reduced, save model parameters
			if model_sse < sse :
				best_pdf = pdf
				sse = model_sse
				best_loc = loc
				best_scale = scale
				best_arg = arg
				best_name = name

			# Defined above an arbitrary threshold (Good enough visual fit)
			if model_sse < sse_thr :
				break
	    
    	# Defined above an arbitrary threshold (Good enough visual fit)
		if not to_break:
			st.subheader("Loi sélectionnée: " + str(best_name))

			df = pd.DataFrame.from_dict({'x':x, 'y':y, 'pdf':best_pdf})

			c2 = alt.Chart(df).mark_line().encode(x='x', y='pdf', color=alt.value("#FFAA00"))
			c1 = alt.Chart(df).mark_circle().encode(x='x', y='y')
			c = c1 + c2
			st.altair_chart(c)

			st.write("Temps d'exécution : ", np.round(time.time() - time_base, 4), " seconds")

			# Give details on the chosen model
			st.write("Loc. : ",  np.round(best_loc,4))
			st.write("Scale. : ", np.round(best_scale, 4))
			st.write("Autres arguments : ", best_arg)
			st.write("SSE : ", sse)
			st.success("Fini!")

if dist == "Gamma":

	shape = st.sidebar.text_input("Shape", "2")
	scale = st.sidebar.text_input("Scale", "1")

	# Génération des données
	try:
		data = np.random.gamma(float(shape),float(scale), length)
		plot_mle(data)

	except ValueError:
		st.error("Entrer un nombre.")


elif dist == "Normal":

	mean = st.sidebar.text_input("Moyenne", "0")
	std = st.sidebar.text_input("Std", "1")

	# Génération des données
	try:
		data = np.random.normal(float(mean),float(std), length)
		plot_mle(data)
		
	except ValueError:
		st.error("Entrer un nombre.")

elif dist == "Lognormal":

	mean = st.sidebar.text_input("Moyenne", "1")
	std = st.sidebar.text_input("Std", "0.8")

	# Génération des données
	try:
		data = np.random.lognormal(float(mean),float(std), length)
		plot_mle(data)
		
	except ValueError:
		st.error("Entrer un nombre.")

