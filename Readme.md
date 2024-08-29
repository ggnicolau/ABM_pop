# Agent-Based Model (ABM) Simulation of Segregation

![ABM Simulation](https://github.com/ggnicolau/ABM_pop/blob/8be7130c80f4a338d05cc153d02510bd32375a13/output/gifs/todos_modelos_combinados.gif)

We have two columns of charts. On the left, the occupied households are represented, and on the right, the occupation of the homeless population. Each row corresponds to a model: the first is the basic model, which includes only households and the homeless population; the second adds the Church agent; and the third adds the Police (in addition to the Church).

## Introduction

In previous research, we used mixed methods, combining ethnography (participant observation and interviews) with data science (exploratory data analysis, regression models, natural language processing, and geostatistics) using computational tools (Python, R, QGIS, Gephi). We collected and analyzed diverse datasets to find evidence of the progressive violence affecting all social aspects of daily life for the homeless population in the city of São Paulo. In this article, we propose an innovative approach that does not rely exclusively on data, utilizing objective logical models to reinforce our previous findings.

## Agent-Based Model (ABM)

The Agent-Based Model (ABM) is a simulation conceived by Nobel Prize winner Thomas Schelling, initially called the "Schelling Model of Segregation." Similar to games like "peg solitaire," Schelling developed a system with rational units, limited space, and a set of rules, resulting in logical outputs without the need for an initial hypothesis. This model simulates much of our social behavior in urban spaces. It is counter-intuitive, as it shows that only a small fraction of intolerant people among a tolerant majority can create entire spaces of segregation. Thus, it proves that the sum of individual rational choices can lead to irrational collective outcomes. Since the 1970s, this discovery has helped shape modern public policies in different cities worldwide to address vicious cycles of segregation and violence in a manner less influenced by moral judgments; furthermore, it is counter-intuitive because it reveals that the most common moral reasoning could intensify the vicious cycle rather than mitigate it.

## Model Development

Based on this approach, we developed our own Agent-Based Model to better understand our object of study. We introduced more agents with different rational behaviors (homeless people, real estate speculators, police, churches, and social services), speculative spatial competition (price), inequality variable (income), and different collective action variables (e.g., violent and non-violent).

## Results

Using our basic simulation model, which involves only two agents (homeless people and property owners), we arrived at a counter-intuitive result that shows no causality between the presence of homeless people and spatial degradation or devaluation. Typically, property owners and tenants move to affordable neighborhoods that become denser and, therefore, richer in resources; once these neighborhoods become expensive and unaffordable, they move to another neighborhood, repeating the cycle of supply and demand. We argue that the homeless population seeks only the services and resources needed to survive and, in response, moves along with other agents who lead the densification process; homelessness, therefore, is not the cause of degradation or devaluation.

As our basic model reaches convergence and stabilizes in a static state of spatial segregation, we observe that when new variables are added, the system's behavior changes significantly. When we include the Church, which attracts and densifies the homeless population in certain areas, the system takes longer to stabilize. The homeless population concentrates in specific areas, while residents disperse and stabilize in more densely occupied surrounding points. Although the system eventually stabilizes, this process occurs more slowly compared to the basic model. However, when we introduce the Police variable, the model begins to converge and stabilize, but it promptly destabilizes and enters a cycle of constant change, never fully converging. Instead, the system becomes highly dynamic and unstable, with incessant displacements of residents, the homeless population, churches, and police, preventing any form of spatial equilibrium.

### Comparison of Model Results

|                         | Basic      | Church     | ChurchPolice  |
|-------------------------|------------|------------|---------------|
| Mobilidade_Morador       | 0.000000   | 0.000000   | 0.805643      |
| Mobilidade_Povo_Rua      | 0.442231   | 0.462151   | 0.645418      |
| Indice_Gini_Morador      | 0.666378   | 0.648766   | 0.223241      |
| Indice_Gini_Povo_Rua     | 0.409044   | 0.417251   | 0.345458      |

### Interpretation of Results

- **Mobilidade_Morador**: The mobility of residents remains zero in both the basic and Church models, indicating stability in resident locations. However, in the ChurchPolice model, the mobility increases significantly, showing that the presence of police forces induces considerable movement among residents, likely due to the introduction of violence and subsequent instability.
  
- **Mobilidade_Povo_Rua**: The mobility of the homeless population increases gradually from the basic model to the Church model and reaches its peak in the ChurchPolice model. This suggests that the presence of the Church already contributes to a slight increase in mobility, but the introduction of police forces leads to even greater displacement, likely due to increased violence and police intervention.
  
- **Indice_Gini_Morador**: The Gini index for residents decreases from the basic model to the ChurchPolice model, indicating that income inequality among residents reduces when both the Church and Police are present. This could be attributed to the forced redistribution of residents caused by police intervention.
  
- **Indice_Gini_Povo_Rua**: The Gini index for the homeless population also decreases slightly across the models, with the lowest value observed in the ChurchPolice model. This reduction indicates a more even distribution of the homeless population, likely due to the combined effects of the Church attracting homeless individuals to specific areas and the police displacing them.

## Conclusion

We hope this simulation can represent the dynamics of urban social inequality. We can push the debate about gentrification beyond moral reasoning or blaming one agent or another during this process and identify intuitive actions which, when taken, intensified the vicious cycle rather than hindered it. We propose this as a step toward focusing on constructive solutions.
