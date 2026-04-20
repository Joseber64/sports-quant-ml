export interface IRepository<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  create(item: Omit<T, 'id'> | T): Promise<T>;
  update(id: string, item: Partial<T>): Promise<T | null>;
  delete(id: string): Promise<boolean>;
}