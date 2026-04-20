import { IRepository } from '../interfaces/IRepository';

export abstract class BaseService<T> {
  constructor(protected readonly repository: IRepository<T>) {}

  async getById(id: string): Promise<T> {
    const entity = await this.repository.findById(id);
    if (!entity) {
      throw new Error('Resource not found');
    }
    return entity;
  }

  async getAll(): Promise<T[]> {
    return this.repository.findAll();
  }
}