La majorité des techniques de fusion, qu’elles soient classiques ou basées sur l’IA, ont été utilisées dans le domaine de la télédétection.

Il faudra réfléchir à l’adaptabilité de notre base de données par rapport aux différentes architectures présentées. Il faudra aussi se demander s’il est nécessaire d’ajouter des blocs de prétraitement et combien d’étapes de prétraitement sont nécessaires. Ces points devront être pris en compte pendant le développement.

Dans un premier temps, l’idée est de développer les modèles de classification, puis d’ajouter au fur et à mesure des blocs de prétraitement supplémentaires selon les besoins identifiés, pour améliorer les performances.

Il faudra également penser à se comparer à d’autres modèles existants, on pourra ainsi se situer en termes de performances par rapport aux techniques de transfert de connaissance qui traitent la fusion des données LiBS avec HSI-NIR (diapo 12).

Les architectures présentées utilisent plutôt une approche de classification à l’échelle du pixel (segmentation). On est dans un cas similaire.

En termes de résolution des données, on est d’environ 200 microns pour le NIR et à 1 mégapixel pour la LiBS.

Un paramètre important à prendre en compte est la taille des images : si elles ne sont pas assez grandes, il est difficile d’aller loin avec les convolutions.

En ce qui concerne la base de données, une grande partie des échantillons est prête (une quarantaine de plots sont prêts, il en reste une dizaine).

Pour la LiBS, on aura environ 1 500 longueurs d’onde pertinentes par détecteur.

Si dans la littérature, les travaux marchent bien avec moins de longueurs d’onde que les nôtres, ça peut poser des limites côté coût de calcul. Comme on a plus de longueurs d’onde, on pourrait être limités de ce côté-là.

Une faible résolution des images peut être un frein pour certaines pistes.

Un bloc d’alignement et de correspondance pixel à pixel est obligatoire si on veut faire de la segmentation. 

On pourra mieux discuter des blocs d’alignement une fois que les données seront entre nos mains et analysées.

On a le choix entre deux options : soit faire des convolutions 3D, mais c’est très lourd, soit ajouter une couche 3D au début de l’extracteur de caractéristiques pour réduire la dimensionnalité, puis continuer avec des convolutions 2D.

Développer des modèles indépendants pour chaque modalité peut nous donner plus d’intuition sur la façon de décider comment fusionner les données.

Pour valoriser la recherche bibliographique, l’objectif est de la soumettre sous forme de revue en se concentrant sur les modèles de fusion. Ce sera une revue de ce qui existe sur la fusion, et non pas un travail de benchmarking.

En termes d’organisation, Marwa, la directrice de thèse, pilotera toute la partie administrative et assurera la relecture des articles. Amine et Aurélie s’occuperont de l’orientation technique sur le Deep Learning et la Computer Vision. Charles sera en charge des analyses élémentaires et apportera son expertise dans le recyclage des plastiques. Loïc gérera la partie technique de la LiBS. Enfin, Marion gérera la partie HSI-NIR et assurera un suivi général sur toute la thèse.

Il a été décidé de faire un point IA avec Marion, Amine et Aurélie toutes les 2 à 3 semaines, et un point avec toute l’équipe toutes les 5 à 6 semaines.

#####################################################################

Faire un état de l’art concernant les métriques
Fournir des informations sur la taille des images (nombre de pixels) pour le HSI-NIR et la LiBS
Vérifier si la volumétrie des données dans les publications est comparable à nos données, en se comparant au nombre de canaux.
Préciser le nombre de longueurs d’onde dans les données HSI-NIR
Voir comment accéder aux données HSI-NIR et comment en extraire les spectres.
Regarder dans la littérature comment les problèmes d’alignement sont traités et comment les auteurs les résolvent.
Faire une veille sur les modèles qui traitent la segmentation d’instances.
Rediscuter de la revue : venir avec des idées et des propositions de plan, et faire les démarches pour identifier et contacter les journaux.
Voir avec Marion pour l’organisation de la réunion de suivi de mars, afin de la programmer le même jour que le séminaire.
Mettre à jour la zone Extranet en déposant les rapports, documents, comptes rendus, articles… (tout au long de la thèse)
Faire une demande pour accéder au supercalculateur Orion.
Voir avec Vincent Le Coq comment s’organiser pour l’utilisation des serveurs R05.
Essayer de voir s’il serait possible de participer au workshop SCAI.
Se renseigner sur les formations imposées par l’école doctorale et sur d’éventuels changements cette année.
Essayer de voir avec l’école doctorale s’il y a des financements possible pour les écoles d’été.
nvoyer un mail a CESI pour vérifier qu’ils acceptent les étudiants en PhD.
Ajouter le CSI sur le Gantt.
Trouver des noms et penser à les solliciter début été.
Rediscuter de la stratégie de rédaction des articles.
Ajouter la labellisation des données dans le Gantt.
Finaliser la préparation des échantillons côté LiBS.