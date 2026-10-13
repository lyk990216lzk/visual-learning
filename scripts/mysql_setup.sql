-- 在 MySQL 中创建数据库与专用账号（字符集必须是 utf8mb4，否则中文校名/类别名会乱码）
-- 用法： mysql -u root -p < scripts/mysql_setup.sql

CREATE DATABASE IF NOT EXISTS image_recognition
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

-- 专用账号（生产环境请把密码改成自己的强密码）
CREATE USER IF NOT EXISTS 'ir_user'@'127.0.0.1' IDENTIFIED BY 'ir_pass_2026';
CREATE USER IF NOT EXISTS 'ir_user'@'localhost' IDENTIFIED BY 'ir_pass_2026';

GRANT ALL PRIVILEGES ON image_recognition.* TO 'ir_user'@'127.0.0.1';
GRANT ALL PRIVILEGES ON image_recognition.* TO 'ir_user'@'localhost';
FLUSH PRIVILEGES;

-- 检查结果
SELECT SCHEMA_NAME AS '已创建数据库', DEFAULT_CHARACTER_SET_NAME AS '字符集'
FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = 'image_recognition';
