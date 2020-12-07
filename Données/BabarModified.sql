DROP TABLE IF EXISTS `high_Degree_Beer`;
CREATE TABLE `high_Degree_Beer` 
	AS SELECT babar_server_purchase.* FROM babar_server_purchase JOIN babar_server_product ON babar_server_purchase.product_id = babar_server_product.id 
WHERE YEAR(TIMESTAMP) > 2013 AND YEAR(TIMESTAMP) < 2020 AND
(NAME = 'Chimay Bleue' OR NAME = 'Chimay Bleu' OR NAME = 'Kwak' OR NAME =
'Karmeliet Triple' OR NAME= 'Duvel' OR NAME = 'Maredsous Triple' OR NAME = 'Chouffe
Pinte' OR NAME = 'Chouffe Demi' OR NAME = 'Triple Karmeliet Pinte' OR NAME = 'Triple
Karmeliet Demi' OR NAME = 'Grim Triple Pinte' OR NAME = 'Grim Triple Demi' OR NAME
= 'bush ambrée' OR NAME = 'Delirium' OR NAME = 'Elephant Pinte' OR NAME = 'Elephant
Demi');


DROP TABLE IF EXISTS `normal_Degree_Beer`;
CREATE TABLE `normal_Degree_Beer` AS SELECT babar_server_purchase.* FROM babar_server_purchase JOIN babar_server_product ON babar_server_purchase.product_id = babar_server_product.id 
WHERE YEAR(TIMESTAMP) > 2013 AND YEAR(TIMESTAMP) < 2020 AND 
(NAME = 'Leffe' OR NAME = 'Grimbergen' OR NAME = 'Kro Demi' OR NAME = 'Kro Pinte'
OR NAME = 'Pelforth' OR NAME = 'Skoll' OR NAME = 'BrewDog Punk IPA' OR NAME =
'Tigre Bock' OR NAME = 'Troll Pinte' OR NAME = 'Troll Demi' OR NAME ='Cuvée des trolls'
OR NAME = 'Paix Dieu 33cL' OR NAME = 'San Miguel' OR NAME = 'Ambrée pinte' OR NAME
= 'Ambrée demi');

DROP TABLE IF EXISTS `Not_Beer`;
CREATE TABLE `Not_Beer` AS SELECT babar_server_purchase.* FROM babar_server_purchase JOIN babar_server_product ON babar_server_purchase.product_id = babar_server_product.id 
WHERE YEAR(TIMESTAMP) > 2013 AND YEAR(TIMESTAMP) < 2020  AND
(NAME = 'Smirnoff' OR NAME = 'Pastis' OR NAME = 'Hard' OR NAME = 'Kir' OR NAME = 'Cocktail Hard' OR NAME = 'Shot' OR NAME = 'Rouge Pinte' OR NAME = 'Rouge Demi'
OR NAME = 'Sangria' OR NAME = 'Granita Hard' OR NAME = 'Hard Qualitée' OR NAME =
'JagerBomb');

DROP TABLE IF EXISTS `Special_Beer`;
CREATE TABLE `Special_Beer` AS SELECT babar_server_purchase.* FROM babar_server_purchase JOIN babar_server_product ON babar_server_purchase.product_id = babar_server_product.id 
WHERE YEAR(TIMESTAMP) > 2013 AND YEAR(TIMESTAMP) < 2020  AND
(NAME = 'Desperados' OR NAME = "Cidre Demi" OR NAME = "Cidre Pinte" OR NAME = "Cidre Doux/Brut" OR NAME ="Cubanisto" OR NAME = "Corona"
OR NAME = "Chimay Rouge" OR NAME = "Hoegaaden blanche" OR NAME ="Chimay Blanche" OR NAME ="Blanche Demi" OR NAME = "Blanche Pinte" OR NAME = "1664 Blanche" OR
NAME = "Pecheresse" OR NAME = "Kriek" OR NAME = "Cherry chouffe" OR NAME = "Delirium rouge pinte");