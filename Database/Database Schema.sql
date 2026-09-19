DROP TABLE IF EXISTS `Titles`;
DROP TABLE IF EXISTS `Friend`;
DROP TABLE IF EXISTS `Users`;
DROP TABLE IF EXISTS `Watched`;

CREATE TABLE `Titles` (
    `show_id` INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `type` VARCHAR(7) NULL,
    `title` VARCHAR(109) NULL,
    `director` VARCHAR(1000) NULL,
    `cast` VARCHAR(1000) NULL,
    `country` VARCHAR(137) NULL,
    `date_added` VARCHAR(19) NULL,
    `release_year` INT NULL,
    `rating` VARCHAR(10) NULL,
    `duration` VARCHAR(10) NULL,
    `listed_in` VARCHAR(79) NULL,
    `description` VARCHAR(1000) NULL,
    `streaming site` ENUM('Netflix','Disney+','Hulu','Prime Video') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE Users (
	user_id INT PRIMARY KEY,
	username VARCHAR(30),
	api_key CHAR(30),
	email VARCHAR(254) NOT NULL,
	password VARCHAR(255) NOT NULL,
	CHECK (CHAR_LENGTH(api_key) = 30)
	);

CREATE TABLE Friend (
	friend_id INT AUTO_INCREMENT PRIMARY KEY,
	user_id_1 INT,
	user_id_2 INT,
	FOREIGN KEY (user_id_1) REFERENCES Users(user_id),
	FOREIGN KEY (user_id_2) REFERENCES Users(user_id)
);

CREATE TABLE Watched (
	review_id INT AUTO_INCREMENT PRIMARY KEY
	user_id INT,
	show_id INT,
	date_watched DATETIME,
	review decimal (2,1)
	description VARCHAR(500).
	FOREIGN KEY user_id REFERENCES Users.user_id,
	FOREIGN KEY show_id REFERENCES Titles.user_id
);
