export default async function handler(req:Request, res:Response) {
    // Handle request logic (e.g., database access)
  
    const data = { message: 'Hello from API!' };
    res.status(200).json(data);
  }