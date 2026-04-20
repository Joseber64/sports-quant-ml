import { Request, Response, NextFunction } from 'express';
import { AppError } from '../errors/AppError';

export const errorHandler = (err: any, req: Request, res: Response, next: NextFunction) => {
  const isAppError = err instanceof AppError;
  const status = isAppError ? err.status : 500;
  const message = isAppError ? err.message : 'Internal Server Error';

  console.error(`[${new Date().toISOString()}] ${err.name}: ${message}`, {
    stack: err.stack,
    path: req.path
  });

  res.status(status).json({
    success: false,
    error: message,
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
    timestamp: new Date().toISOString()
  });
};