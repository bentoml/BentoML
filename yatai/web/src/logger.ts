import * as winston from "winston";

const logger = winston.createLogger({
  levels: winston.config.npm.levels,
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
});

export const createLogger = (base_log_path: string) => {
  logger.add(new winston.transports.File({ filename: base_log_path }));
};

export const getLogger = () => {
  return logger;
};
