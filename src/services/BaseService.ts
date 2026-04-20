import { IRepository } from '../interfaces/IRepository';
import { NotFoundError } from '../errors/AppError';

export abstract class BaseService<T> {
  constructor(protected readonly repository: IRepository<T>) {}

  async getById(id: string): Promise<T> {
    const entity = await this.repository.findById(id);
    if (!entity) {
      throw new NotFoundError();
    }
    return entity;
  }

  async getAll(): Promise<T[]> {
    return this.repository.findAll();
  }

  async create(data: T): Promise<T> {
    return this.repository.create(data);
  }

  async update(id: string, data: Partial<T>): Promise<T> {
    const updated = await this.repository.update(id, data);
    if (!updated) {
      throw new NotFoundError();
    }
    return updated;
  }

  async delete(id: string): Promise<void> {
    const deleted = await this.repository.delete(id);
    if (!deleted) {
      throw new NotFoundError();
    }
  }
}