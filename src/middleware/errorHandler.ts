import { Request, Response, NextFunction } from 'express';

export const errorHandler = (err: any, req: Request, res: Response, next: NextFunction) => {
  const status = err.status || 500;
  const message = err.message || 'Internal Server Error';

  console.error(`[Error]: ${message}`, { stack: err.stack });

  res.status(status).json({
    success: false,
    error: message,
    timestamp: new Date().toISOString()
  });
};